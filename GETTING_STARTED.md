# Getting Started with Axon

Axon has five entry points that all share the same knowledge base. Ingest once, query from anywhere.

![Axon entry points](docs/assets/diagrams/entry-points.png)

---

## Quick Install

```bash
git clone https://github.com/jyunming/Axon.git && cd Axon
pip install -e .

# Pull a local LLM (pick one)
ollama pull llama3.1:8b   # recommended — 4.7 GB, ~8 GB RAM
ollama pull phi3:mini     # minimal  — 2.3 GB, ~4 GB RAM
```

## Install the VS Code Extension

Install the bundled extension: open Extensions panel (`Ctrl+Shift+X`) → `···` → **Install from VSIX** → select `integrations/vscode-axon/axon-copilot-1.0.0.vsix` → reload VS Code.

> Full setup guide including Python discovery, settings, and troubleshooting: [SETUP.md § 11](SETUP.md#11-vs-code-extension-github-copilot-integration)

**How the extension finds Python** (for `autoStart` — starting `axon-api` automatically):

| Your install method | What to do |
|---|---|
| `pip install` into a venv | Run `axon` once from the terminal → auto-detected via `~/.axon/.python_path` |
| `pipx install axon` | Nothing — extension finds the pipx venv automatically |
| Workspace venv (`.venv/`) | Nothing — extension checks the open folder automatically |
| Custom / unusual path | Set `axon.pythonPath` in VS Code Settings (Ctrl+,) |

No configuration needed for the most common cases. If auto-detection fails, VS Code shows a notification linking directly to the setting.

---

## How Ingestion Works

![Ingestion flow](docs/assets/diagrams/ingestion-flow.png)

## How Querying Works

![Query flow](docs/assets/diagrams/query-flow.png)

---

## Entry Point 1 — CLI / REPL

**Launch:**
```bash
axon
```

**Ingest:**
```
/ingest ./my-documents/       # ingest a folder
/ingest ./report.pdf          # ingest a single file
```

**Query:**
```
You: What are the main topics in these documents?
You: Summarise the Q3 report
You: Explain this code @./src/main.py
```

**Useful commands:**

| Command | What it does |
|---|---|
| `/ingest <path>` | Ingest file or folder |
| `/list` | Show all ingested documents |
| `/model <name>` | Switch LLM on the fly (`llama3.1:8b`, `gemini-1.5-flash`, `gpt-4o`) |
| `/project switch <name>` | Change knowledge base |
| `/rag topk 10` | Retrieve more chunks |
| `/rag rerank` | Toggle BGE reranker |
| `/rag hyde` | Toggle HyDE query expansion |
| `/llm temperature 0.2` | Set LLM temperature (0.0–2.0) |
| `/sessions` | Browse saved conversation sessions |
| `/context` | Show current config and token usage |
| `/clear` | Start a fresh conversation |
| `/help` | Full command list |

---

## Entry Point 2 — Web UI

**Launch:**
```bash
axon-ui   # opens http://localhost:8501
```

**Ingest:**
- Left sidebar → **Knowledge Hub** → paste a URL, enter a path, or upload a file

**Query:**
- Type in the chat input → Enter

**Settings:**
- Left sidebar → **Model & Settings** → toggle hybrid search, reranking, HyDE, RAPTOR, GraphRAG, temperature

---

## Entry Point 3 — REST API

**Launch:**
```bash
axon-api   # starts at http://localhost:8000
```

**Full interactive reference:** `http://localhost:8000/docs`

**Ingest (async):**
```bash
# Start ingest — returns job_id immediately
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/docs"}'
# → {"job_id": "abc123", "status": "processing"}

# Poll until complete
curl http://localhost:8000/ingest/status/abc123
# → {"status": "completed", "documents_ingested": 12}
```

**Ingest text directly:**
```bash
curl -X POST http://localhost:8000/add_text \
  -H "Content-Type: application/json" \
  -d '{"text": "Important note to remember.", "metadata": {"source": "notes"}}'
```

**Query (with answer synthesis):**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?"}'
```

**Search (raw chunks, no LLM):**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication flow", "top_k": 5}'
```

---

## Entry Point 4 — MCP Server (Copilot Agent Mode)

The MCP server gives GitHub Copilot direct tool access in **agent mode** (hammer icon in Copilot Chat).

![MCP workflow](docs/assets/diagrams/mcp-workflow.png)

**Setup (one-time):** Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "axon": {
      "type": "stdio",
      "command": "axon-mcp",
      "env": {
        "RAG_API_BASE": "http://localhost:8000"
      }
    }
  }
}
```

Create `.vscode/settings.json`:
```json
{
  "chat.mcp.access": "all"
}
```

**Launch:**
```bash
axon-api   # must be running
```
Reload VS Code → Copilot agent mode → Axon tools appear automatically.

> **Windows:** if `axon-mcp` is not on PATH, use the full path: `C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts\axon-mcp.exe`

> Full MCP setup including Linux/WSL paths, team sharing, and Windows workarounds: [SETUP.md § 10](SETUP.md#10-mcp-setup-copilot-agent-mode)

---

## Entry Point 5 — VS Code Extension (Copilot Chat)

After installing the VSIX and starting `axon-api`, use Copilot Chat (Ctrl+Shift+I):

![VS Code extension workflow](docs/assets/diagrams/vscode-workflow.png)

**Ingest:**
```
Ingest my documents at /path/to/docs
Add this URL to my knowledge base: https://docs.example.com
```

**Query:**
```
Search my knowledge base for information about the login flow.
What does the authentication module do?
```

**Manage:**
```
List all my projects
Switch to the "work" project
What files have I ingested?
```

**Image ingest** (requires GPT-4o or Claude model in Copilot):
```
Describe and ingest this diagram: /path/to/architecture.png
```

**Graph panel** (see [Show Graph Panel](#visualize-graphs--vs-code-graph-panel-recommended) above):
```
Axon: Show Graph for Query…        ← Command Palette
Axon: Show Graph for Selection     ← Command Palette, with text selected

@workspace show me the graph for how retrieval works
@workspace visualise the authentication module
```

---

## Projects — Multiple Knowledge Bases

Isolate documents by project. Parent projects automatically search all children.

![Projects hierarchy](docs/assets/diagrams/projects-hierarchy.png)

```bash
# REPL
/project new research/papers
/project switch work
/project list

# CLI
axon --project work "Summarise the Q3 report"
axon --project research "What papers discuss attention mechanisms?"

# API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics?", "project": "work"}'
```

---

---

## RAPTOR + GraphRAG — Best-Known Method

RAPTOR and GraphRAG are **on by default** as of v1.0.0. They make Axon much better at multi-hop
and corpus-wide questions, but they add LLM calls during ingest. Here is how to get the most out
of them without paying unnecessary cost.

### What each feature does

| Feature | What it adds | Ingest cost |
|---|---|---|
| **RAPTOR** | Hierarchical summaries — groups ~5 leaf chunks per source into a summary node; summary nodes are retrieved alongside leaf chunks for multi-hop questions | ~1 LLM call per 5 chunks |
| **GraphRAG (light — default)** | Entity graph via regex noun-phrase extraction; zero LLM calls; populates the VS Code Graph Panel KG tab; entity co-occurrence used to expand retrieval | **Zero LLM calls** |
| **GraphRAG (standard)** | Adds LLM entity descriptions + typed relation triples; richer graph, better retrieval expansion | ~1–3 LLM calls per chunk |
| **RAPTOR + GraphRAG** | Auto-composition — large sources use RAPTOR summaries as GraphRAG extraction units, cutting costs by ~50–80% | Shared cost — not additive |

### Default config: fast graph, no extra LLM calls

The default `config.yaml` ships with **fast graph mode**:

```yaml
rag:
  graph_rag: true
  graph_rag_depth: light      # regex noun-phrase extraction — zero LLM calls
  graph_rag_relations: false  # relation extraction (LLM-heavy) is off
  graph_rag_community: false  # community detection is off
```

This gives you the VS Code Graph Panel KG tab for free, with no ingest slowdown. Ingest speed is the same as having `graph_rag: false`.

### Upgrading to a richer graph

When you want typed entities with descriptions and relation edges (at the cost of LLM calls per chunk):

```yaml
rag:
  graph_rag_depth: standard   # LLM entity descriptions
  graph_rag_relations: true   # SUBJECT | RELATION | OBJECT triples
```

### Further reducing ingest time

**Skip RAPTOR for large sources:**
```yaml
rag:
  raptor_max_source_size_mb: 2.0   # skip RAPTOR for sources > 2 MB
```

**Disable both for bulk ingest, enable for daily use:**
```yaml
rag:
  raptor: false
  graph_rag: false
```

### Visualize graphs — VS Code Graph Panel (recommended)

The **Axon Graph Panel** opens a live split-view directly inside VS Code. No browser, no extra tools needed.

```
Command Palette (Ctrl+Shift+P) → Axon: Show Graph for Query…
```

Or select a block of code / text, then:

```
Command Palette → Axon: Show Graph for Selection
```

Or from Copilot Chat:

```
@workspace visualise how the retrieval pipeline works
@workspace show me the graph for authentication
```

**What the panel shows:**

```
┌──────────────────────┬──────────────────────────────────────┐
│  Q: How does         │  [ Knowledge Graph ]  [ Code Graph ] │
│     retrieval work?  │                                      │
│  ────────────────    │   ●─────────────◆                   │
│  LLM answer with     │      3D force-graph                  │
│  inline citations    │   ▼              ▼                   │
│  ────────────────    │   ●              ●                   │
│  [1] retrievers.py ▸ │                                      │
│  [2] main.py:142  ▸  │   click node → jump to source file  │
└──────────────────────┴──────────────────────────────────────┘
```

| Tab | What it shows | Requires |
|---|---|---|
| **Knowledge Graph** | Entity–relation graph extracted from **any document** (PDF, DOCX, Markdown…) | `graph_rag: true` — **on by default**, just ingest your docs |
| **Code Graph** | File/class/function structure with import/call edges for source code | `code_graph: true` in `config.yaml` (opt-in) |

Disabled tabs show a tooltip explaining which flag to enable. Both tabs can be present in the same panel.

Clicking any **citation** or **graph node** opens the associated source file at the exact line in the editor.

### Visualize graphs — HTML export (REPL / headless)

For offline use or sharing, export the entity graph as a self-contained HTML file:

**REPL:**
```
/graph-viz                        # saves to temp dir and prints the path
/graph-viz /path/to/output.html   # saves to a specific file
```

**API:**
```bash
curl http://localhost:8000/graph/visualize -o graph.html
# Open graph.html in any browser — no server required
```

Requires `pyvis`: `pip install axon[graphrag]`

---

## Where to go next

| Guide | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | Full platform install, all model options, VS Code extension config, MCP setup |
| [QUICKREF.md](QUICKREF.md) | All CLI flags, REPL commands, API endpoints at a glance |
| [MODEL_GUIDE.md](MODEL_GUIDE.md) | Choosing an LLM and embedding model for your hardware |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors and fixes |
