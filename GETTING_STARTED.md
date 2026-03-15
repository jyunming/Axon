# Getting Started with Axon

Axon is a local RAG (Retrieval-Augmented Generation) assistant. You point it at documents, and it lets you ask questions about them — using an LLM running on your own machine.

This guide covers the core ideas and the most common workflows.

---

## The core idea

LLMs are good at reasoning but they don't know your documents.
RAG fixes that by:

1. **Ingesting** your documents — splitting them into chunks and storing them as vector embeddings
2. **Retrieving** relevant chunks when you ask a question
3. **Answering** using the retrieved text as context

Axon does all three, locally, with no data leaving your machine.

---

## Install

```bash
pip install -e ".[dev]"
```

You also need [Ollama](https://ollama.ai/) running with at least one model pulled:

```bash
ollama pull gemma
```

---

## The three entry points

| Command | What it does |
|---|---|
| `axon` | Interactive REPL — best for exploring |
| `axon-api` | FastAPI server — use when integrating with other tools |
| `axon-ui` | Streamlit web UI — visual chat interface |

For day-to-day use, `axon` is the starting point.

---

## Ingest documents, then ask questions

```bash
# Start the REPL
axon

# Inside the REPL, ingest a folder
/ingest ./my-documents/

# Ask a question
You: What is the main topic of these documents?
```

You can also attach files inline without ingesting:

```
You: Explain this code @./src/axon/main.py
You: What changed in @./src/axon/
```

---

## Projects — keeping knowledge bases separate

A **project** is an isolated knowledge base. Documents ingested into one project are not visible in another.

```bash
axon --project work "Summarise the Q3 report"
axon --project personal "What did I write about sleep?"
```

You can nest projects up to 5 levels deep:

```
research
research/papers
research/papers/2024
```

When you switch to a parent project (e.g. `research`), Axon automatically searches across all its sub-projects too.

```bash
# Inside the REPL
/project new research/papers
/project switch research       # searches research + all children
/project list                  # shows the full tree
```

---

## Switching models at runtime

```bash
# Switch to a different Ollama model
/model llama3.1:8b

# Switch to a cloud model (needs API key)
/model gemini-1.5-flash
/model gpt-4o
```

---

## Useful REPL commands

| Command | Purpose |
|---|---|
| `/help` | Full command reference |
| `/ingest <path>` | Add documents to the current project |
| `/list` | Show all ingested documents |
| `/model` | Switch LLM |
| `/rag topk 5` | Retrieve fewer chunks (faster, less context) |
| `/context` | Show current settings and token usage |
| `/sessions` | Browse saved sessions |
| `/clear` | Start a fresh conversation |

---

## Configuration

Copy `config.yaml` and edit it to set your default model, embedding provider, and storage paths.

The most commonly changed settings:

```yaml
llm:
  model: gemma        # any Ollama model you have pulled

rag:
  top_k: 10           # max chunks passed to the LLM (retrieval may fetch more internally for reranking)
  hybrid_search: true # combine vector + keyword search

# Custom storage location (optional)
# projects_root: /path/to/your/projects
```

---

## Where to go next

- `QUICKREF.md` — complete command reference
- `SETUP.md` — detailed installation and configuration options
- `TROUBLESHOOTING.md` — common issues and fixes
- `MODEL_GUIDE.md` — choosing models for different use cases
