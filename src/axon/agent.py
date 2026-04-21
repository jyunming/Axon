"""
src/axon/agent.py

Shared agentic loop for the CLI REPL and Streamlit web GUI.

Provides:
  - REPL_TOOLS: OpenAI-format tool schemas for in-session agent use.
  - ToolCall: lightweight namedtuple returned by complete_with_tools().
  - dispatch_tool(): executes a tool call against an AxonBrain instance.
  - run_agent_loop(): drives the multi-turn tool-call loop.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import namedtuple
from collections.abc import Callable

logger = logging.getLogger("Axon")

# ---------------------------------------------------------------------------
# ToolCall — returned by OpenLLM.complete_with_tools()
# ---------------------------------------------------------------------------

ToolCall = namedtuple("ToolCall", ["name", "args", "thought_signature"], defaults=[None])

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI / Ollama function-calling format)
# ---------------------------------------------------------------------------

REPL_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "ingest_path",
            "description": (
                "Ingest a local file, directory, or glob pattern into the knowledge base. "
                "Use this when the user asks to add, load, index, or ingest documents from disk. "
                "If the user specifies a target project, pass it as 'project' — the tool will "
                "switch to that project automatically before ingesting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to a file, directory, or glob (e.g. ./docs, ./src/*.py).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to ingest into. If omitted, uses the currently active project.",
                    },
                    "force": {
                        "type": "boolean",
                        "description": (
                            "If true, bypass deduplication and re-ingest even if the content was previously seen. "
                            "Use this when list_knowledge shows the file is missing but ingest says it was already ingested."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_knowledge",
            "description": (
                "List all indexed sources with chunk counts. "
                "Use this when the user asks what is in the knowledge base, what documents are indexed, etc. "
                "If the user specifies a target project, pass it as 'project'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Optional project to list. If omitted, uses the currently active project.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Search the knowledge base and return relevant raw document chunks. "
                "Use this for retrieval before synthesizing an answer. "
                "If the user specifies a target project, pass it as 'project'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project to search in. If omitted, uses the currently active project.",
                    },
                    "filters": {
                        "type": "object",
                        "description": (
                            "Optional metadata key/value pairs to filter results "
                            '(e.g. {"source": "report.pdf"}).'
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_text",
            "description": (
                "Add a short text snippet or note directly to the knowledge base. "
                "Use this only for text the user dictates or pastes inline — "
                "NOT for attached files (use ingest_path for those). "
                "If the user specifies a target project, pass it as 'project'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text content to save.",
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional label / source identifier for this text (e.g. filename or URL).",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to save into. If omitted, uses the currently active project.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "purge_source",
            "description": (
                "Remove stale content-hash records and optionally delete all vectors for a given source. "
                "Use this when ingest_path says 'deduplication skipped' but list_knowledge shows the file "
                "is NOT in the knowledge base — purge_source clears the stale hashes so the file can be "
                "re-ingested normally. This is a destructive operation — always confirm with the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": (
                            "Absolute path of the file whose stale hashes should be removed. "
                            "Must be the same absolute path that was (or would be) used by ingest_path."
                        ),
                    },
                    "also_delete_vectors": {
                        "type": "boolean",
                        "description": (
                            "If true (default), also delete any existing vector chunks for this source "
                            "before clearing hashes. Set to false to only remove hashes."
                        ),
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_documents",
            "description": (
                "Remove documents from the knowledge base by their source name. "
                "This is a destructive operation — always confirm with the user before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source name to delete (e.g. path or label used at ingest time).",
                    }
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_project",
            "description": (
                "Wipe ALL documents from the active project. "
                "This is irreversible — always confirm with the user before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all available Axon projects.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_project",
            "description": "Switch the active project to a different named project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Project name to switch to.",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Create a new named Axon project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name for the new project (lowercase, alphanumeric, hyphens, slashes for hierarchy).",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional human-readable description for the project.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_config",
            "description": "Return all active AxonConfig fields and the active project name.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_url",
            "description": (
                "Fetch an HTTP/HTTPS URL and ingest its text content into the knowledge base. "
                "Use this whenever the user provides a URL to ingest — NEVER use ingest_path for URLs. "
                "Private/internal URLs (127.x, 10.x, 192.168.x) are blocked for security."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The HTTP or HTTPS URL to fetch and ingest.",
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project name to ingest into. If omitted, uses the currently active project.",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_settings",
            "description": (
                "Update active RAG configuration settings for the current session. "
                "Use this when the user asks to change top_k, enable/disable reranking, "
                "HyDE, hybrid search, GraphRAG, or other RAG flags. "
                "Show the before/after values in your response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to retrieve (1-50).",
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score (0.0-1.0).",
                    },
                    "hybrid_search": {
                        "type": "boolean",
                        "description": "Enable BM25 + vector hybrid search.",
                    },
                    "rerank": {"type": "boolean", "description": "Enable cross-encoder reranking."},
                    "hyde": {
                        "type": "boolean",
                        "description": "Enable HyDE (hypothetical document embeddings).",
                    },
                    "multi_query": {
                        "type": "boolean",
                        "description": "Enable multi-query expansion.",
                    },
                    "step_back": {"type": "boolean", "description": "Enable step-back prompting."},
                    "query_decompose": {
                        "type": "boolean",
                        "description": "Enable query decomposition.",
                    },
                    "compress_context": {
                        "type": "boolean",
                        "description": "Enable context compression.",
                    },
                    "graph_rag": {
                        "type": "boolean",
                        "description": "Enable GraphRAG entity expansion.",
                    },
                    "raptor": {
                        "type": "boolean",
                        "description": "Enable RAPTOR hierarchical summarisation.",
                    },
                    "truth_grounding": {
                        "type": "boolean",
                        "description": "Enable web search grounding.",
                    },
                    "discussion_fallback": {
                        "type": "boolean",
                        "description": "Enable discussion fallback.",
                    },
                    "sentence_window": {
                        "type": "boolean",
                        "description": "Enable sentence-window retrieval.",
                    },
                    "sentence_window_size": {
                        "type": "integer",
                        "description": "Sentence window size (1-10).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_texts",
            "description": (
                "Batch-ingest multiple text snippets into the knowledge base in a single call. "
                "Prefer this over calling add_text repeatedly when saving several snippets at once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "docs": {
                        "type": "array",
                        "description": "List of text snippets to ingest.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Text content."},
                                "source": {
                                    "type": "string",
                                    "description": "Optional source label.",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Optional extra metadata.",
                                },
                            },
                            "required": ["text"],
                        },
                    },
                    "project": {
                        "type": "string",
                        "description": "Optional project to save into. If omitted, uses the currently active project.",
                    },
                },
                "required": ["docs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stale_docs",
            "description": (
                "Return documents that have not been re-ingested within a given number of days. "
                "Useful for identifying outdated knowledge that should be refreshed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Flag documents not refreshed within this many days (default 30).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_project",
            "description": (
                "Permanently delete a project and ALL its indexed data. "
                "This is irreversible — always confirm with the user before calling."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the project to delete.",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_status",
            "description": (
                "Return the current GraphRAG knowledge-graph status: entity count, "
                "relation count, community summary count, and whether a rebuild is in progress."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "graph_finalize",
            "description": (
                "Trigger a GraphRAG community rebuild. Use after ingesting many documents "
                "when GraphRAG is enabled and you want to refresh community summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refresh_ingest",
            "description": (
                "Re-ingest all tracked file sources whose content has changed on disk. "
                "Skips files that have not changed. Use this to keep the knowledge base up to date."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Optional project name. If omitted, uses the currently active project.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a bash shell command and return its stdout and stderr. "
                "Use this to run system commands, inspect files, run scripts, check git status, "
                "search file contents, list directories, etc. "
                "Commands run through bash (auto-detected: Git Bash or WSL on Windows, "
                "native bash on Linux/macOS; falls back to cmd.exe if bash is unavailable). "
                "The working directory matches the REPL's current cwd. "
                "Avoid destructive commands (rm -rf, git reset --hard, etc.) unless the user "
                "explicitly requested them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "The bash command to execute "
                            "(e.g. 'ls -la', 'git status', 'cat README.md', 'grep -r TODO src/')."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30, max 120).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file. "
                "Use this instead of bash heredoc whenever you need to write multi-line text, "
                "code snippets, markdown, or any content that contains special shell characters "
                "(curly braces, backticks, dollar signs, backslashes, etc.). "
                "Accepts absolute paths (e.g. C:\\\\Users\\\\name\\\\file.md) and relative paths. "
                "Default mode overwrites the file; use mode='a' to append."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write to the file.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "'w' to overwrite (default) or 'a' to append.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read and return the text content of a file. "
                "Use this to inspect a file's current contents before editing, "
                "or to confirm a write_file operation succeeded. "
                "Accepts absolute and relative paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file to read.",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

# Destructive tools that require confirmation before execution.
# run_shell is included so the webapp never exposes arbitrary shell execution.
_DESTRUCTIVE_TOOLS = {
    "purge_source",
    "delete_documents",
    "clear_project",
    "delete_project",
    "run_shell",
}

# Webapp uses a non-destructive subset — delete/clear must use sidebar controls.
WEBAPP_TOOLS = [t for t in REPL_TOOLS if t["function"]["name"] not in _DESTRUCTIVE_TOOLS]

# ---------------------------------------------------------------------------
# dispatch_tool — executes a single tool call against AxonBrain
# ---------------------------------------------------------------------------


def dispatch_tool(
    brain,
    tool_name: str,
    args: dict,
    *,
    confirm_cb: Callable[[str], bool] | None = None,
) -> str:
    """Execute a single tool call and return a human-readable result string.

    Args:
        brain: AxonBrain instance.
        tool_name: Name of the tool to run (must be in REPL_TOOLS).
        args: Tool arguments parsed from the LLM response.
        confirm_cb: Optional callable(message) -> bool for destructive ops.
            If None, destructive tools are blocked.

    Returns:
        A string summary of the tool result, suitable for feeding back to the LLM.
    """
    if tool_name in _DESTRUCTIVE_TOOLS:
        if confirm_cb is None:
            return (
                f"⚠️  Tool '{tool_name}' requires confirmation but no confirm_cb is set — skipped."
            )
        msg = {
            "purge_source": f"Remove stale hashes and vectors for source '{args.get('source', '?')}' from the knowledge base?",
            "delete_documents": f"Delete all documents with source '{args.get('source', '?')}' from the knowledge base?",
            "clear_project": f"Wipe ALL documents from project '{brain._active_project}'? This is irreversible.",
            "delete_project": f"Permanently delete project '{args.get('name', '?')}' and ALL its data? This is irreversible.",
            "run_shell": f"Run shell command: {args.get('command', '?')!r}?",
        }.get(tool_name, f"Run {tool_name}?")
        if not confirm_cb(msg):
            return f"🚫 '{tool_name}' cancelled by user."

    try:
        if tool_name == "ingest_path":
            return _tool_ingest_path(brain, args)
        elif tool_name == "purge_source":
            return _tool_purge_source(brain, args)
        elif tool_name == "list_knowledge":
            return _tool_list_knowledge(brain, args)
        elif tool_name == "search_knowledge":
            return _tool_search_knowledge(brain, args)
        elif tool_name == "add_text":
            return _tool_add_text(brain, args)
        elif tool_name == "delete_documents":
            return _tool_delete_documents(brain, args)
        elif tool_name == "clear_project":
            return _tool_clear_project(brain)
        elif tool_name == "list_projects":
            return _tool_list_projects()
        elif tool_name == "switch_project":
            return _tool_switch_project(brain, args)
        elif tool_name == "create_project":
            return _tool_create_project(brain, args)
        elif tool_name == "get_config":
            return _tool_get_config(brain)
        elif tool_name == "ingest_url":
            return _tool_ingest_url(brain, args)
        elif tool_name == "update_settings":
            return _tool_update_settings(brain, args)
        elif tool_name == "ingest_texts":
            return _tool_ingest_texts(brain, args)
        elif tool_name == "get_stale_docs":
            return _tool_get_stale_docs(brain, args)
        elif tool_name == "delete_project":
            return _tool_delete_project(brain, args)
        elif tool_name == "graph_status":
            return _tool_graph_status(brain)
        elif tool_name == "graph_finalize":
            return _tool_graph_finalize(brain)
        elif tool_name == "refresh_ingest":
            return _tool_refresh_ingest(brain, args)
        elif tool_name == "run_shell":
            return _tool_run_shell(args)
        elif tool_name == "write_file":
            return _tool_write_file(args)
        elif tool_name == "read_file":
            return _tool_read_file(args)
        else:
            return f"Unknown tool: '{tool_name}'"
    except Exception as exc:
        logger.debug("dispatch_tool %s error: %s", tool_name, exc)
        return f"⚠️  Tool '{tool_name}' failed: {exc}"


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------


def _make_vision_fn(brain):
    """Return a vision callable ``(image_bytes: bytes) -> str`` backed by brain.llm,
    or None if vision OCR is disabled or the LLM does not support it."""
    if brain is None:
        return None
    if not getattr(getattr(brain, "config", None), "pdf_vision_ocr", True):
        return None
    llm = getattr(brain, "llm", None)
    if llm is None or not callable(getattr(llm, "complete_with_image", None)):
        return None

    _ocr_prompt = (
        "You are an OCR engine. Extract ALL text from this image exactly as it appears. "
        "Preserve paragraph structure. Output only the extracted text with no commentary."
    )

    def _vision_fn(image_bytes: bytes) -> str:
        try:
            return llm.complete_with_image(_ocr_prompt, image_bytes) or ""
        except Exception as exc:
            logger.warning("Vision OCR LLM call failed: %s", exc)
            return ""

    return _vision_fn


def _tool_ingest_path(brain, args: dict) -> str:
    import glob as _glob

    from axon.loaders import DirectoryLoader

    path = args.get("path", "").strip()
    if not path:
        return "No path provided."

    logger.info(
        "ingest_path called: path=%r project=%r cwd=%r", path, args.get("project"), os.getcwd()
    )

    # B1: Route URLs to URLLoader rather than the file-system glob
    if path.startswith("http://") or path.startswith("https://"):
        return _tool_ingest_url(brain, {"url": path, "project": args.get("project", "")})

    # Expand ~ and resolve relative paths: try CWD first, then home directory.
    # This handles cases where the user typed @Downloads/file.pdf (relative to home)
    # and the LLM passed it without the leading ~.
    from pathlib import Path as _Path

    _p = _Path(path).expanduser()
    if not _p.is_absolute():
        _cwd_try = _Path(os.getcwd()) / path
        if _cwd_try.exists():
            path = str(_cwd_try)
            logger.debug("ingest_path resolved to cwd: %r", path)
        else:
            _home_try = _Path.home() / path
            if _home_try.exists():
                path = str(_home_try)
                logger.info("ingest_path resolved to home fallback: %r", path)
    else:
        path = str(_p)

    project = args.get("project", "").strip()
    if project:
        brain.switch_project(project)

    brain._assert_write_allowed("ingest")

    matched = sorted(_glob.glob(path, recursive=True))
    logger.info("ingest_path glob matched %d file(s): %s", len(matched), matched[:5])
    if not matched:
        if os.path.isdir(path):
            matched = [path]
        else:
            logger.warning("ingest_path: no files matched %r (cwd=%r)", path, os.getcwd())
            return f"No files matched: {path}"

    force_reingest = bool(args.get("force", False))
    _vision_fn = _make_vision_fn(brain)
    loader_mgr = DirectoryLoader(vision_fn=_vision_fn)
    ingested, skipped, total_chunks, dedup_skipped, empty_content = 0, 0, 0, 0, 0

    def _ingest_docs(docs):
        """Run brain.ingest(), bypassing dedup when force=True."""
        if force_reingest:
            orig = brain.config.dedup_on_ingest
            brain.config.dedup_on_ingest = False
            try:
                return brain.ingest(docs)
            finally:
                brain.config.dedup_on_ingest = orig
        return brain.ingest(docs)

    for p in matched:
        if os.path.isdir(p):
            docs = loader_mgr.load(p)
            if docs:
                n = _ingest_docs(docs)
                if n == 0:
                    dedup_skipped += 1
                else:
                    total_chunks += n
                    ingested += 1
            else:
                ingested += 1  # empty dir counts as visited
        elif os.path.isfile(p):
            ext = os.path.splitext(p)[1].lower()
            if ext in loader_mgr.loaders:
                docs = loader_mgr.loaders[ext].load(p)
                if not docs:
                    # Loader returned nothing — either the file is corrupt, unreadable,
                    # or image-based (e.g., scanned PDF with no extractable text).
                    empty_content += 1
                else:
                    non_empty = [d for d in docs if d.get("text", "").strip()]
                    if not non_empty:
                        # File loaded but every page/record has empty text — most commonly
                        # a scanned/image-based PDF that needs OCR to extract content.
                        logger.warning(
                            "ingest_path: '%s' loaded %d doc(s) but all have empty text "
                            "(image-based PDF or unreadable format?)",
                            p,
                            len(docs),
                        )
                        empty_content += 1
                    else:
                        n = _ingest_docs(docs)
                        if n == 0:
                            dedup_skipped += 1
                        else:
                            total_chunks += n
                            ingested += 1
            else:
                skipped += 1

    if ingested == 0 and empty_content > 0:
        return (
            f"⚠️ {empty_content} file(s) loaded but contained no extractable text. "
            f"This usually means the file is an image-based or scanned PDF that requires OCR. "
            f"Try converting the PDF to text first, or use a document with embedded text."
        )

    if ingested == 0 and skipped > 0:
        return f"No supported files ingested (skipped {skipped} unsupported file type(s))."

    active_project = brain._active_project

    if ingested == 0 and dedup_skipped > 0:
        return (
            f"⚠️ All {dedup_skipped} file(s) were already in the knowledge base "
            f"(deduplication skipped them). Use list_knowledge to see existing documents. "
            f"If the file is missing from list_knowledge, re-run with force=true to bypass deduplication."
        )

    result = (
        f"Ingested {ingested} file(s) / director{'y' if ingested == 1 else 'ies'} "
        f"({total_chunks} chunk(s)), skipped {skipped} (unsupported) "
        f"into project '{active_project}'."
    )
    if dedup_skipped > 0:
        result += f" ({dedup_skipped} file(s) skipped — already ingested.)"
    if empty_content > 0:
        result += f" ({empty_content} file(s) skipped — no extractable text.)"
    return result


def _tool_list_knowledge(brain, args: dict) -> str:
    project = args.get("project", "").strip() if args else ""
    prev = brain._active_project if project else None
    if project:
        brain.switch_project(project)
    try:
        docs = brain.list_documents()
        if not docs:
            return "Knowledge base is empty."
        total_chunks = sum(d["chunks"] for d in docs)
        lines = [f"{len(docs)} source(s), {total_chunks} chunk(s) total:"]
        for d in docs:
            lines.append(f"  {d['source']}  ({d['chunks']} chunks)")
        return "\n".join(lines)
    finally:
        if prev is not None and prev != brain._active_project:
            brain.switch_project(prev)


def _tool_search_knowledge(brain, args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "No query provided."
    top_k = int(args.get("top_k", 5))

    project = args.get("project", "").strip()
    prev = brain._active_project if project else None
    if project:
        brain.switch_project(project)

    filters = args.get("filters") or {}
    overrides: dict = {"top_k": top_k}
    if filters:
        overrides["where"] = filters

    try:
        results, _diag, _trace = brain.search_raw(query, overrides=overrides)
    except Exception as exc:
        return f"Search failed: {exc}"
    finally:
        if prev is not None and prev != brain._active_project:
            brain.switch_project(prev)

    if not results:
        return "No results found."
    lines = [f"{len(results)} result(s) for '{query}':"]
    for i, r in enumerate(results, 1):
        src = r.get("metadata", {}).get("source", r.get("id", "?"))
        score = r.get("score", 0)
        snippet = r.get("text", "")[:200].replace("\n", " ")
        lines.append(f"  [{i}] {src} (score={score:.3f}): {snippet}…")
    return "\n".join(lines)


def _tool_add_text(brain, args: dict) -> str:
    import uuid

    from axon.loaders import SmartTextLoader

    project = args.get("project", "").strip()
    if project:
        brain.switch_project(project)

    brain._assert_write_allowed("ingest")

    text = args.get("text", "").strip()
    if not text:
        return "No text provided."
    source = args.get("source") or f"agent_note_{uuid.uuid4().hex[:8]}"
    loader = SmartTextLoader()
    docs = loader.load_text(text, source=source)
    brain.ingest(docs)
    active_project = brain._active_project
    return f"Saved {len(docs)} chunk(s) with source '{source}' into project '{active_project}'."


def _tool_purge_source(brain, args: dict) -> str:
    """Remove stale content hashes and optionally vectors for a source path."""
    brain._assert_write_allowed("purge_source")

    source = args.get("source", "").strip()
    if not source:
        return "No source provided."

    also_delete_vectors = bool(args.get("also_delete_vectors", True))
    removed_vectors = 0
    removed_hashes = 0

    # Step 1: Delete vectors if requested (and source is in the doc index)
    if also_delete_vectors:
        all_docs = brain.list_documents()
        target = [d for d in all_docs if d["source"] == source]
        if target:
            ids_to_delete = [cid for d in target for cid in d.get("doc_ids", [])]
            if ids_to_delete:
                brain.vector_store.delete_by_ids(ids_to_delete)
                if brain.bm25 is not None:
                    brain.bm25.delete_documents(ids_to_delete)
                if brain._entity_graph:
                    brain._prune_entity_graph(set(ids_to_delete))
                removed_vectors = len(ids_to_delete)

    # Step 2: Recompute chunk hashes from the source file and remove from hash store
    import os

    from axon.loaders import DirectoryLoader

    abs_path = source if os.path.isabs(source) else os.path.abspath(source)
    if not os.path.exists(abs_path):
        for candidate in [source, os.path.join(os.getcwd(), source)]:
            if os.path.exists(candidate):
                abs_path = os.path.abspath(candidate)
                break

    if os.path.exists(abs_path):
        try:
            loader_mgr = DirectoryLoader(vision_fn=_make_vision_fn(brain))
            raw_docs = loader_mgr.load(abs_path)
            chunks = []
            for doc in raw_docs:
                dt, has_code = brain._detect_dataset_type(doc)
                doc.setdefault("metadata", {})["dataset_type"] = dt
                sp = brain._get_splitter_for_type(dt, has_code, source=abs_path)
                if sp is None:
                    sp = brain.splitter
                if sp:
                    chunks.extend(sp.transform_documents([doc]))
                else:
                    chunks.append(doc)
            for chunk in chunks:
                h = brain._doc_hash(chunk)
                if h in brain._ingested_hashes:
                    brain._ingested_hashes.discard(h)
                    removed_hashes += 1
            if removed_hashes:
                brain._save_hash_store()
        except Exception as e:
            logger.debug("purge_source hash removal error: %s", e)

    parts = []
    if removed_vectors:
        parts.append(f"deleted {removed_vectors} vector chunk(s)")
    if removed_hashes:
        parts.append(f"purged {removed_hashes} stale content hash(es)")

    if not parts:
        return (
            f"⚠️ Nothing found to purge for source '{source}'. "
            "Verify the source name matches exactly what list_knowledge shows, "
            "and that the file path is accessible."
        )

    return f"✓ Purged '{source}': {', '.join(parts)}. " "You can now re-ingest the file normally."


def _tool_delete_documents(brain, args: dict) -> str:
    brain._assert_write_allowed("delete")

    source = args.get("source", "").strip()
    if not source:
        return "No source provided."

    all_docs = brain.list_documents()
    target = [d for d in all_docs if d["source"] == source]
    if not target:
        return f"No documents found with source '{source}'."

    ids_to_delete = [cid for d in target for cid in d.get("doc_ids", [])]
    if not ids_to_delete:
        return f"Source '{source}' found but has no chunk IDs."

    # Delete from vector store
    brain.vector_store.delete_by_ids(ids_to_delete)
    # Delete from BM25 index
    if brain.bm25 is not None:
        brain.bm25.delete_documents(ids_to_delete)
    # Prune entity graph (no-op if graph is empty)
    if brain._entity_graph:
        brain._prune_entity_graph(set(ids_to_delete))

    # Purge content hashes so the source can be re-ingested without force=true.
    # The hashes are keyed by chunk content, not by ID, so we retrieve the chunk
    # texts from the doc index metadata if available, else use the stored IDs as
    # a best-effort lookup.
    removed_hashes = 0
    for d in target:
        for cid in d.get("doc_ids", []):
            # Each stored ID encodes the hash as the last segment (id format: source::hash).
            parts = cid.rsplit("::", 1)
            if len(parts) == 2:
                h = parts[1]
                if h in brain._ingested_hashes:
                    brain._ingested_hashes.discard(h)
                    removed_hashes += 1
    if removed_hashes:
        try:
            brain._save_hash_store()
        except Exception:
            pass

    note = f" (purged {removed_hashes} content hash(es))" if removed_hashes else ""
    return f"Deleted {len(ids_to_delete)} chunk(s) from source '{source}'.{note}"


def _tool_clear_project(brain) -> str:
    from axon.collection_ops import clear_active_project

    brain._assert_write_allowed("clear")

    project = brain._active_project
    clear_active_project(brain)
    return f"Cleared all documents from project '{project}'."


def _tool_list_projects() -> str:
    from axon.projects import list_projects

    projects = list_projects()
    if not projects:
        return "No projects found."
    lines = [f"{len(projects)} project(s):"]
    for p in projects:
        name = p.get("name", "?")
        desc = p.get("description", "")
        children = p.get("children", [])
        child_names = ", ".join(c.get("name", "") for c in children) if children else ""
        line = f"  {name}"
        if desc:
            line += f"  — {desc}"
        if child_names:
            line += f"  [sub: {child_names}]"
        lines.append(line)
    return "\n".join(lines)


def _tool_switch_project(brain, args: dict) -> str:
    name = args.get("name", "").strip()
    if not name:
        return "No project name provided."
    brain.switch_project(name)
    return f"Switched to project '{name}'."


def _tool_create_project(brain, args: dict) -> str:
    from axon.projects import ensure_project

    name = args.get("name", "").strip()
    if not name:
        return "No project name provided."
    description = args.get("description", "").strip()
    ensure_project(name, description=description)
    brain.switch_project(name)
    return f"Project '{name}' created." + (f" Description: {description}" if description else "")


def _tool_get_config(brain) -> str:
    import dataclasses

    cfg = brain.config
    lines = [f"active_project: {brain._active_project}"]
    for field in dataclasses.fields(cfg):
        lines.append(f"{field.name}: {getattr(cfg, field.name)}")
    return "\n".join(lines)


def _tool_ingest_url(brain, args: dict) -> str:
    from axon.loaders import URLLoader

    url = args.get("url", "").strip()
    if not url:
        return "No URL provided."
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"Invalid URL — must start with http:// or https://: {url}"

    project = args.get("project", "").strip()
    if project:
        brain.switch_project(project)

    brain._assert_write_allowed("ingest")

    try:
        docs = URLLoader().load(url)
    except Exception as exc:
        return f"Failed to fetch '{url}': {exc}"

    if not docs:
        return f"No content extracted from '{url}'."

    brain.ingest(docs)
    active_project = brain._active_project
    return f"Ingested {len(docs)} chunk(s) from '{url}' into project '{active_project}'."


_UPDATABLE_SETTINGS = {
    "top_k": int,
    "similarity_threshold": float,
    "hybrid_search": bool,
    "rerank": bool,
    "hyde": bool,
    "multi_query": bool,
    "step_back": bool,
    "query_decompose": bool,
    "compress_context": bool,
    "graph_rag": bool,
    "raptor": bool,
    "truth_grounding": bool,
    "discussion_fallback": bool,
    "sentence_window": bool,
    "sentence_window_size": int,
}


def _tool_update_settings(brain, args: dict) -> str:
    cfg = brain.config
    changes: list[str] = []
    for field, cast in _UPDATABLE_SETTINGS.items():
        if field in args and args[field] is not None:
            old_val = getattr(cfg, field, None)
            new_val = cast(args[field])
            if old_val != new_val:
                setattr(cfg, field, new_val)
                changes.append(f"  {field}: {old_val!r} → {new_val!r}")
    if not changes:
        return "No settings changed (all values already match or no valid fields provided)."
    return "Updated settings:\n" + "\n".join(changes)


def _tool_ingest_texts(brain, args: dict) -> str:
    import uuid

    from axon.loaders import SmartTextLoader

    items = args.get("docs", [])
    if not items:
        return "No text snippets provided."

    project = args.get("project", "").strip()
    if project:
        brain.switch_project(project)

    brain._assert_write_allowed("ingest")

    loader = SmartTextLoader()
    all_docs: list[dict] = []
    for item in items:
        text = item.get("text", "").strip()
        if not text:
            continue
        source = item.get("source") or f"agent_note_{uuid.uuid4().hex[:8]}"
        extra_meta = item.get("metadata", {}) or {}
        for d in loader.load_text(text, source=source):
            d["metadata"].update(extra_meta)
            all_docs.append(d)

    if not all_docs:
        return "All provided snippets were empty."

    brain.ingest(all_docs)
    active_project = brain._active_project
    return (
        f"Ingested {len(all_docs)} chunk(s) from {len(items)} snippet(s) "
        f"into project '{active_project}'."
    )


def _tool_get_stale_docs(brain, args: dict) -> str:
    from datetime import datetime, timezone

    days = int(args.get("days", 30))
    doc_versions = getattr(brain, "_doc_versions", {}) or {}
    if not doc_versions:
        return "No ingestion history tracked in this session."

    cutoff = datetime.now(timezone.utc).timestamp() - days * 86_400
    stale: list[str] = []
    for source, info in doc_versions.items():
        if not isinstance(info, dict):
            continue
        ingested_str = info.get("ingested_at", "")
        try:
            ts = datetime.fromisoformat(ingested_str.replace("Z", "+00:00")).timestamp()
        except (ValueError, AttributeError):
            continue
        if ts < cutoff:
            age = round((datetime.now(timezone.utc).timestamp() - ts) / 86_400, 1)
            stale.append(f"  {source}  (last ingested {age} day(s) ago)")

    if not stale:
        return f"No stale documents found (threshold: {days} day(s))."
    return f"{len(stale)} stale source(s) (>{days} days):\n" + "\n".join(stale)


def _tool_delete_project(brain, args: dict) -> str:
    from axon.projects import delete_project

    name = args.get("name", "").strip()
    if not name:
        return "No project name provided."
    try:
        delete_project(name)
    except Exception as exc:
        return f"Failed to delete project '{name}': {exc}"
    return f"Project '{name}' and all its data have been permanently deleted."


def _tool_graph_status(brain) -> str:
    entity_count = len(getattr(brain, "_entity_graph", {}) or {})
    relation_count = len(getattr(brain, "_relation_graph", {}) or {})
    community_count = len(getattr(brain, "_community_summaries", {}) or {})
    in_progress = getattr(brain, "_community_build_in_progress", False)
    dirty = getattr(brain, "_community_graph_dirty", False)
    lines = [
        f"entities: {entity_count}",
        f"relations: {relation_count}",
        f"community summaries: {community_count}",
        f"rebuild in progress: {in_progress}",
        f"graph dirty (rebuild needed): {dirty}",
    ]
    return "\n".join(lines)


def _tool_graph_finalize(brain) -> str:
    try:
        brain.finalize_ingest()
    except Exception as exc:
        return f"Graph finalize failed: {exc}"
    community_count = len(getattr(brain, "_community_summaries", {}) or {})
    return f"Graph finalize complete. Community summaries: {community_count}."


def _tool_refresh_ingest(brain, args: dict) -> str:
    import hashlib
    import os as _os

    from axon.loaders import DirectoryLoader

    project = args.get("project", "").strip()
    if project:
        brain.switch_project(project)

    brain._assert_write_allowed("ingest")

    doc_versions = getattr(brain, "_doc_versions", {}) or {}
    if not doc_versions:
        return "No ingestion history tracked — nothing to refresh."

    loader_mgr = DirectoryLoader(vision_fn=_make_vision_fn(brain))
    refreshed, skipped, errors = 0, 0, 0

    for source, info in list(doc_versions.items()):
        if not isinstance(info, dict):
            continue
        if not _os.path.isfile(source):
            continue

        stored_hash = info.get("content_hash", "")
        try:
            ext = _os.path.splitext(source)[1].lower()
            if ext not in loader_mgr.loaders:
                skipped += 1
                continue
            docs = loader_mgr.loaders[ext].load(source)
            combined = "".join(d.get("text", "") for d in docs)
            current_hash = hashlib.sha256(combined.encode("utf-8", errors="replace")).hexdigest()
            if current_hash == stored_hash:
                skipped += 1
                continue
            brain.ingest(docs)
            refreshed += 1
        except Exception as exc:
            logger.debug("refresh_ingest error for %s: %s", source, exc)
            errors += 1

    return (
        f"Refresh complete: {refreshed} file(s) re-ingested, "
        f"{skipped} unchanged, {errors} error(s)."
    )


def _tool_write_file(args: dict) -> str:
    from pathlib import Path

    path = Path(args.get("path", "").strip()).expanduser()
    content = args.get("content", "")
    mode = args.get("mode", "w")
    if mode not in ("w", "a"):
        mode = "w"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "w":
            path.write_text(content, encoding="utf-8")
        else:
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        verb = "Wrote" if mode == "w" else "Appended"
        return f"{verb} {len(content)} characters to {path}"
    except Exception as exc:
        return f"⚠️  write_file failed: {exc}"


def _tool_read_file(args: dict) -> str:
    from pathlib import Path

    path = Path(args.get("path", "").strip()).expanduser()
    max_chars = 8000
    try:
        text = path.read_text(encoding="utf-8")
        if len(text) > max_chars:
            return text[:max_chars] + f"\n… (truncated — {len(text)} total chars)"
        return text
    except Exception as exc:
        return f"⚠️  read_file failed: {exc}"


def _tool_run_shell(args: dict) -> str:
    """Execute a bash command and return stdout + stderr as a string."""
    import subprocess

    from axon.repl import _find_bash

    command = args.get("command", "").strip()
    if not command:
        return "⚠️  run_shell: no command provided."

    timeout = min(int(args.get("timeout", 30)), 120)

    # All output uses plain print + ANSI — routes through patch_stdout correctly
    # from the agent thread. rich Console buffers and does not go through the hook.
    _YEL = "\x1b[1;33m"  # bold yellow — Bash label
    _GRN = "\x1b[1;32m"  # bold green  — success
    _YLW = "\x1b[1;33m"  # bold yellow — warning (stderr)
    _RED = "\x1b[1;31m"  # bold red    — failure
    _DIM = "\x1b[2m"
    _RST = "\x1b[0m"

    print(f"\n  {_YEL}Bash{_RST}  {_DIM}$ {command}{_RST}")

    def _p(line: str, color: str = "") -> None:
        print(f"{color}{line}{_RST}" if color else line)

    bash = _find_bash()
    try:
        result = subprocess.run(
            [*bash, command] if bash else command,
            shell=not bool(bash),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
            encoding="utf-8",
            errors="replace",
        )
        success = result.returncode == 0
        has_stderr = bool(result.stderr.strip())
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if has_stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        if not success:
            parts.append(f"[exit {result.returncode}]")
        elif not result.stdout and not has_stderr:
            parts.append("(exit 0 — command succeeded, no output)")
        output = "\n".join(parts)

        if not success:
            icon, clr = "✗", _RED
        elif has_stderr:
            icon, clr = "⚠", _YLW
        else:
            icon, clr = "✓", _GRN
        lines = output.splitlines()
        if lines:
            _p(f"  {icon}  {lines[0]}", clr)
            for line in lines[1:]:
                _p(f"     {line}")
        else:
            _p(f"  {icon}", clr)

        sys.stdout.flush()
        return output
    except subprocess.TimeoutExpired:
        _msg = f"Command timed out after {timeout}s."
        _p(f"  ✗  {_msg}", _RED)
        return f"⚠️  {_msg}"
    except Exception as exc:
        _msg = f"run_shell failed: {exc}"
        _p(f"  ✗  {_msg}", _RED)
        return f"⚠️  {_msg}"


# ---------------------------------------------------------------------------
# run_agent_loop — multi-turn tool-calling loop
# ---------------------------------------------------------------------------

_AGENT_SYSTEM_PROMPT = """\
You are Axon Agent, an AI assistant with access to a local knowledge-base (RAG system).
You can manage the knowledge base and answer questions about it using the available tools.

## Ingestion decision tree
To ingest content, choose the right tool:
- URL starting with http:// or https:// → use `ingest_url` (NEVER use ingest_path for URLs)
- Local file path or glob (e.g. ./docs, *.py) → use `ingest_path`
- Multiple text snippets to save at once → use `ingest_texts`
- Single short note the user dictates → use `add_text`

## Project handling
- NEVER call `switch_project` unless the user explicitly asks to change projects.
- Ingest, search, and list always operate on the currently active project by default.
- If the user specifies a target project for ingest/search/list, pass it as the `project`
  argument directly — do NOT call `switch_project` separately first.
- Call `get_config` when you need to confirm which project is currently active.
- When the message begins with '[Attached file path(s): ...]', always use `ingest_path`
  with those exact file paths.

## Settings changes
- When the user asks to change top_k, enable/disable reranking, HyDE, hybrid search,
  GraphRAG, or other RAG flags, use `update_settings`. Always show the before/after values.

## Verification
- After ingesting, call `list_knowledge` to confirm documents were indexed.
- After switching or creating a project, confirm with `list_projects` if unsure.

## File operations
- To write multi-line text, code, or markdown to a file: ALWAYS use `write_file`, never bash heredoc.
  Heredoc via bash -c is unreliable for content with special characters (braces, backticks, $, slashes).
- To read a file's contents: use `read_file`.
- When the user's message contains `@file` references, the resolved absolute path(s) appear in a
  `[Attached file path(s): ...]` header at the top of the message. ALWAYS use those absolute paths
  when calling `ingest_path` — never use the relative form from the user's original text.
  - If `ingest_path` reports "deduplication skipped" AND `list_knowledge` shows the file is NOT in the
    knowledge base, the file has stale content hashes. Fix this by:
    1. Calling `purge_source(source=<absolute_path>)` to remove stale hashes (and any orphaned vectors).
    2. Then calling `ingest_path` again normally (without force=true).
    Do NOT use force=true first — purge_source is the correct fix that also updates hash tracking.
    Do NOT retry more than once after purging. If it still fails, report the error to the user.
  - If `ingest_path` reports "no extractable text" or "image-based" or "scanned", the PDF
    is image-only. Vision OCR is handled automatically when the LLM supports it — if it still
    fails after one retry, report the error to the user. STOP retrying further.
    Do NOT call purge_source for this case.
  - NEVER call the same tool with the same arguments more than once. If a tool returns the same error
    twice, stop retrying and explain the situation to the user instead.
  - Do NOT write <tool_call> or <tool_calls> XML in your response text. Tool calls are handled via the
    function-calling API. Only use text to communicate with the user.
- After writing, confirm the path and character count from the tool result.

## Shell commands
- After run_shell completes, report what was done (file created, command output, etc.).
- Never ask the user for content, filenames, or paths that were already stated in their message.
- exit 0 with no output means the command succeeded — confirm that and move on.
- Only run a shell command once; do not retry a succeeded command.

## Tool discipline
- Only call a tool when needed — for general questions or conversation, respond directly.
- If multiple steps are needed, perform them one at a time.
- After each tool call, use the result to inform your next step or final response.
"""


def run_agent_loop(
    llm,
    brain,
    prompt: str,
    chat_history: list[dict],
    *,
    tools: list[dict] | None = None,
    confirm_cb: Callable[[str], bool] | None = None,
    step_cb: Callable[[str, str], None] | None = None,
    max_steps: int = 8,
) -> str:
    """Run the agentic tool-calling loop and return the final text response.

    Args:
        llm: OpenLLM instance.
        brain: AxonBrain instance.
        prompt: Current user message.
        chat_history: Conversation history (list of role/content dicts).
        tools: Tool schemas to expose (defaults to REPL_TOOLS).
        confirm_cb: Callable(message) -> bool for destructive tool confirmation.
        step_cb: Optional callable(tool_name, result) called after each tool step.
            Surfaces can use this to render intermediate progress cards.
        max_steps: Maximum tool-call iterations before forcing a plain text response.

    Returns:
        Final assistant text response.
    """
    active_tools = tools if tools is not None else REPL_TOOLS
    messages = list(chat_history)
    original_prompt = prompt
    current_prompt = prompt

    # Track (tool_name, frozen_args) pairs to detect identical retry loops.
    # If the same tool is called with the same arguments twice in a row, stop.
    _last_call_sig: str | None = None
    _identical_retries = 0

    for _step in range(max_steps):
        result = llm.complete_with_tools(
            current_prompt,
            active_tools,
            system_prompt=_AGENT_SYSTEM_PROMPT,
            chat_history=messages,
        )

        if isinstance(result, str):
            # Plain text response — strip any leaked XML or JSON tool-call syntax and return
            import re as _re

            result = _re.sub(r"<tool_calls?>.*?</tool_calls?>", "", result, flags=_re.DOTALL)
            result = _re.sub(r'\{[\s]*"tool_calls"\s*:.*\}', "", result, flags=_re.DOTALL).strip()
            return result

        # result is a list of ToolCall namedtuples
        tool_results: list[str] = []
        tool_calls_raw: list[tuple] = []
        for tc in result:
            # Build a signature for this call to detect identical retries
            import json as _json

            _sig = f"{tc.name}:{_json.dumps(tc.args, sort_keys=True)}"
            if _sig == _last_call_sig:
                _identical_retries += 1
                if _identical_retries >= 2:
                    # LLM is stuck in a retry loop — break and ask it to summarise
                    logger.debug(
                        "Agent loop: detected identical retry for %s, forcing exit", tc.name
                    )
                    return llm.complete(
                        f"You tried the same tool call ({tc.name}) multiple times with the same arguments "
                        f"and got the same result each time. Please summarise what you found and explain "
                        f"any limitations to the user without retrying. Original request: {original_prompt}",
                        system_prompt=_AGENT_SYSTEM_PROMPT,
                        chat_history=messages,
                    )
            else:
                _last_call_sig = _sig
                _identical_retries = 0

            tool_result = dispatch_tool(brain, tc.name, tc.args, confirm_cb=confirm_cb)
            logger.debug("Agent tool %s(%s) → %s", tc.name, tc.args, tool_result[:100])
            tool_calls_raw.append(
                (tc.name, tc.args, tool_result, getattr(tc, "thought_signature", None))
            )
            tool_results.append(f"[{tc.name}]: {tool_result}")
            if step_cb is not None:
                try:
                    step_cb(tc.name, tool_result)
                except Exception:
                    pass

        # Feed tool results back; keep original user intent visible.
        # __tool_calls__ stores raw (name, args, result) for providers (e.g. Gemini) that
        # need native function_call / function_response parts in their conversation history.
        tool_summary = "\n".join(tool_results)
        messages = messages + [
            {"role": "user", "content": original_prompt},
            {
                "role": "assistant",
                "content": f"<tool_results>\n{tool_summary}\n</tool_results>",
                "__tool_calls__": tool_calls_raw,
            },
        ]
        current_prompt = (
            f"The tool calls above completed. Original request: {original_prompt}\n\n"
            "Confirm to the user what was accomplished. "
            "Do NOT ask for information that was already provided in the original request."
        )

    # Exhausted max_steps — ask the LLM to summarise without tools
    final = llm.complete(
        f"Please summarise what was accomplished and provide a final response. "
        f"Original request: {original_prompt}",
        system_prompt=_AGENT_SYSTEM_PROMPT,
        chat_history=messages,
    )
    return final
