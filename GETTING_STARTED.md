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

## Where to go next

| Guide | What it covers |
|---|---|
| [SETUP.md](SETUP.md) | Full platform install, all model options, VS Code extension config, MCP setup |
| [QUICKREF.md](QUICKREF.md) | All CLI flags, REPL commands, API endpoints at a glance |
| [MODEL_GUIDE.md](MODEL_GUIDE.md) | Choosing an LLM and embedding model for your hardware |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common errors and fixes |
