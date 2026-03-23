"""CLI entry point and project-tree helpers for Axon."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("Axon")


# ---------------------------------------------------------------------------
# Shared helpers used by both CLI (main()) and REPL
# ---------------------------------------------------------------------------


def _print_project_tree(proj_list: list, active: str, indent: int = 0) -> None:
    """Print a recursive project tree with active-project marker and metadata.

    Uses the already-fetched ``children`` list from list_projects() rather than
    calling has_children() again, avoiding redundant directory traversals.
    """
    pad = "  " * indent
    for p in proj_list:
        marker = "●" if p["name"] == active else " "
        ts = p["created_at"][:10] if p["created_at"] else ""
        desc = f"  {p.get('description', '')}" if p.get("description") else ""
        merged = "  [merged]" if p.get("children") else ""
        short = p["name"].split("/")[-1]
        state = p.get("maintenance_state", "normal")
        maint = f"  [{state}]" if state != "normal" else ""
        print(f"  {pad}{marker} {short:<22} {ts}{merged}{maint}{desc}")
        _print_project_tree(p.get("children", []), active, indent + 1)


def _write_python_discovery() -> None:
    """Write current Python executable path to ~/.axon/.python_path.

    Called once at startup so the VS Code extension can auto-detect the Python
    interpreter regardless of whether axon was installed via pip, venv, or pipx.
    Failures are silently ignored — this is a best-effort helper.
    """
    try:
        discovery_dir = Path.home() / ".axon"
        discovery_dir.mkdir(parents=True, exist_ok=True)
        (discovery_dir / ".python_path").write_text(sys.executable, encoding="utf-8")
    except Exception:
        pass


def main():
    import argparse

    from axon.config import AxonConfig
    from axon.main import AxonBrain
    from axon.repl import _infer_provider, _InitDisplay, _interactive_repl

    # On Windows, switch the console to UTF-8 (codepage 65001) so that
    # box-drawing characters and emoji render correctly.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _write_python_discovery()
    parser = argparse.ArgumentParser(description="Axon CLI")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--ingest", help="Path to file or directory to ingest")
    parser.add_argument(
        "--list", action="store_true", help="List all ingested sources in the knowledge base"
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Project to use (must exist; use --project-new to create). "
        'Use "default" for the global knowledge base.',
    )
    parser.add_argument(
        "--project-new",
        metavar="NAME",
        help="Create a new project (if it does not exist) and use it. "
        "Combine with --ingest to populate in one step.",
    )
    parser.add_argument("--project-list", action="store_true", help="List all projects and exit")
    parser.add_argument(
        "--project-delete", metavar="NAME", help="Delete a project and all its data, then exit"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: ~/.config/axon/config.yaml)",
    )
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "ollama_cloud", "openai", "vllm", "github_copilot"],
        help="LLM provider to use (overrides config)",
    )
    parser.add_argument(
        "--model",
        help="Model name to use (overrides config), e.g. gemma:2b, gemini-1.5-flash, gpt-4o",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and supported cloud providers",
    )
    parser.add_argument(
        "--pull", metavar="MODEL", help="Pull an Ollama model by name, e.g. --pull gemma:2b"
    )
    parser.add_argument(
        "--embed",
        metavar="MODEL",
        help="Embedding model to use, e.g. all-MiniLM-L6-v2 or ollama/nomic-embed-text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="F",
        help="LLM temperature for generation (0.0–2.0, default: from config, usually 0.7). "
        "Lower = more deterministic, higher = more creative.",
    )
    parser.add_argument(
        "--discuss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable discussion fallback (answer from general knowledge when no docs match). "
        "Use --discuss to enable, --no-discuss to disable.",
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Brave web search fallback (requires BRAVE_API_KEY). "
        "Use --search to enable, --no-search to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        metavar="N",
        help="Number of chunks to retrieve (default: from config, usually 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        metavar="F",
        help="Similarity threshold for retrieval, 0.0–1.0 (default: from config, usually 0.3)",
    )
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable hybrid BM25+vector search",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable cross-encoder reranking",
    )
    parser.add_argument(
        "--reranker-model",
        metavar="MODEL",
        help="Re-ranker model to use (e.g. BAAI/bge-reranker-v2-m3 for SOTA accuracy, "
        "default: cross-encoder/ms-marco-MiniLM-L-6-v2)",
    )
    parser.add_argument(
        "--hyde",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable HyDE (hypothetical document embedding)",
    )
    parser.add_argument(
        "--multi-query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable multi-query retrieval",
    )
    parser.add_argument(
        "--step-back",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable step-back prompting (abstracts query before retrieval)",
    )
    parser.add_argument(
        "--decompose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable query decomposition (breaks complex questions into sub-questions)",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable LLM context compression (extracts only relevant sentences before generation)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable in-memory query result caching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip LLM; run retrieval only and print diagnostics + ranked chunks (requires --query)",
    )
    parser.add_argument(
        "--raptor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RAPTOR hierarchical summarisation nodes during ingest",
    )
    parser.add_argument(
        "--raptor-group-size",
        type=int,
        metavar="N",
        help="Number of consecutive chunks to group per RAPTOR summary (default: 5)",
    )
    parser.add_argument(
        "--graph-rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable GraphRAG entity-centric retrieval expansion",
    )
    parser.add_argument(
        "--cite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable inline [Document N] citations in the response (default: from config, usually on)",
    )
    parser.add_argument(
        "--code-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable structural code-symbol graph construction during ingest "
        "(requires code files; stored in .code_graph.json)",
    )
    parser.add_argument(
        "--code-graph-bridge",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Phase-3 code-graph bridge: link prose chunks that mention code "
        "symbols to their definition nodes (requires --code-graph)",
    )
    parser.add_argument(
        "--sentence-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable sentence-window retrieval (returns surrounding sentences for each matched chunk)",
    )
    parser.add_argument(
        "--sentence-window-size",
        type=int,
        metavar="N",
        help="Number of surrounding sentences to include per chunk (1-10, requires --sentence-window)",
    )
    parser.add_argument(
        "--crag-lite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable CRAG-Lite (lightweight corrective RAG: grades retrieved docs, "
        "falls back to web search when confidence is low)",
    )
    parser.add_argument(
        "--graph-rag-mode",
        choices=["local", "global", "hybrid"],
        help="GraphRAG traversal mode: local=neighbours only, global=community summaries, "
        "hybrid=both (requires --graph-rag)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-ingest any tracked files whose content has changed since last ingest, then exit",
    )
    parser.add_argument(
        "--list-stale",
        action="store_true",
        help="List documents not re-ingested within --stale-days days, then exit",
    )
    parser.add_argument(
        "--stale-days",
        type=int,
        default=7,
        metavar="N",
        help="Age threshold in days for --list-stale (default: 7)",
    )
    parser.add_argument(
        "--graph-status",
        action="store_true",
        help="Print knowledge graph status (entity count, code nodes, community state), then exit",
    )
    parser.add_argument(
        "--graph-finalize",
        action="store_true",
        help="Rebuild community summaries and finalize the knowledge graph, then exit",
    )
    parser.add_argument(
        "--graph-export",
        action="store_true",
        help="Export the entity graph as an HTML file and print its path, then exit",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable ingest deduplication (allow re-ingesting identical content)",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=["recursive", "semantic"],
        help="Chunking strategy for ingest (recursive or semantic)",
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        metavar="N",
        help="Enable small-to-big retrieval: index child chunks of --chunk-size tokens "
        "but return parent passages of N tokens as LLM context. 0 = disabled.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress spinners and progress (auto-enabled when stdin is not a TTY)",
    )
    # ── Document deletion ────────────────────────────────────────────────────
    parser.add_argument(
        "--delete-doc",
        metavar="SOURCE",
        help="Delete all chunks for a source (match by source path/name), then exit",
    )
    parser.add_argument(
        "--delete-doc-id",
        nargs="+",
        metavar="ID",
        help="Delete specific chunk IDs directly (space-separated), then exit",
    )
    # ── Store init ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--store-init",
        metavar="PATH",
        help="Initialise AxonStore multi-user mode at PATH (e.g. ~/axon_data), then exit",
    )
    # ── Share lifecycle ──────────────────────────────────────────────────────
    parser.add_argument(
        "--share-list",
        action="store_true",
        help="List shares issued by and received by this user, then exit",
    )
    parser.add_argument(
        "--share-generate",
        nargs=2,
        metavar=("PROJECT", "GRANTEE"),
        help="Generate a read-only share key for PROJECT and GRANTEE, then exit",
    )
    parser.add_argument(
        "--share-redeem",
        metavar="SHARE_STRING",
        help="Redeem a share string and mount the shared project, then exit",
    )
    parser.add_argument(
        "--share-revoke",
        metavar="KEY_ID",
        help="Revoke a previously issued share by KEY_ID, then exit",
    )
    # ── Sessions ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--session-list",
        action="store_true",
        help="List saved conversation sessions for the current project, then exit",
    )
    args = parser.parse_args()

    # Suppress httpx INFO noise before _InitDisplay is active (ollama.list fires early)
    if sys.stdin.isatty():
        logging.getLogger("httpx").propagate = False
        logging.getLogger("httpx").setLevel(logging.WARNING)

    config = AxonConfig.load(args.config)
    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        _PROVIDERS = ("ollama", "gemini", "openai", "ollama_cloud", "vllm", "github_copilot")
        if "/" in args.model:
            _prov, _mdl = args.model.split("/", 1)
            if _prov in _PROVIDERS:
                config.llm_provider = _prov
                config.llm_model = _mdl
            else:
                # Not a provider prefix — treat whole string as model name
                config.llm_provider = _infer_provider(args.model)
                config.llm_model = args.model
        else:
            config.llm_provider = _infer_provider(args.model)
            config.llm_model = args.model

    if args.embed:
        _EMBED_PROVIDERS = ("sentence_transformers", "ollama", "fastembed", "openai")
        if "/" in args.embed:
            _eprov, _emdl = args.embed.split("/", 1)
            if _eprov in _EMBED_PROVIDERS:
                config.embedding_provider = _eprov
                config.embedding_model = _emdl
            else:
                config.embedding_model = args.embed
        else:
            config.embedding_model = args.embed

    if args.temperature is not None:
        config.llm_temperature = args.temperature
    if args.discuss is not None:
        config.discussion_fallback = args.discuss
    if args.search is not None:
        config.truth_grounding = args.search
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.hybrid is not None:
        config.hybrid_search = args.hybrid
    if args.rerank is not None:
        config.rerank = args.rerank
    if args.reranker_model:
        config.reranker_model = args.reranker_model
    if args.chunk_strategy:
        config.chunk_strategy = args.chunk_strategy
    if args.parent_chunk_size is not None:
        config.parent_chunk_size = args.parent_chunk_size
    if args.hyde is not None:
        config.hyde = args.hyde
    if args.multi_query is not None:
        config.multi_query = args.multi_query
    if args.step_back is not None:
        config.step_back = args.step_back
    if args.decompose is not None:
        config.query_decompose = args.decompose
    if args.compress is not None:
        config.compress_context = args.compress
    if args.cache is not None:
        config.query_cache = args.cache
    if args.no_dedup:
        config.dedup_on_ingest = False
    if args.raptor is not None:
        config.raptor = args.raptor
    if args.raptor_group_size is not None:
        config.raptor_chunk_group_size = args.raptor_group_size
    if args.graph_rag is not None:
        config.graph_rag = args.graph_rag
    if args.cite is not None:
        config.cite = args.cite
    if args.code_graph is not None:
        config.code_graph = args.code_graph
    if args.code_graph_bridge is not None:
        config.code_graph_bridge = args.code_graph_bridge
    if args.sentence_window is not None:
        config.sentence_window = args.sentence_window
    if args.sentence_window_size is not None:
        config.sentence_window_size = args.sentence_window_size
    if args.crag_lite is not None:
        config.crag_lite = args.crag_lite
    if args.graph_rag_mode is not None:
        config.graph_rag_mode = args.graph_rag_mode

    if args.list_models:
        print("\n  Supported LLM providers and example models:\n")
        print("  ollama       (local)  — gemma:2b, gemma, llama3.1, mistral, phi3")
        print("  gemini       (cloud)  — gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash")
        print("  ollama_cloud (cloud)  — any model hosted at your OLLAMA_CLOUD_URL")
        print("  openai       (cloud)  — gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
        print(
            "  vllm         (local)  — any model served by vLLM (e.g., meta-llama/Llama-3.1-8B-Instruct)"
        )
        print(
            f"               URL: {config.vllm_base_url}  (set vllm_base_url in config or VLLM_BASE_URL env)\n"
        )
        try:
            import ollama as _ollama

            response = _ollama.list()
            models = response.models if hasattr(response, "models") else response.get("models", [])
            if models:
                print("  Locally available Ollama models:")
                for m in models:
                    name = m.model if hasattr(m, "model") else m.get("name", str(m))
                    size_gb = m.size / 1e9 if hasattr(m, "size") and m.size else 0
                    size_str = f"  ({size_gb:.1f} GB)" if size_gb else ""
                    print(f"     • {name}{size_str}")
        except Exception:
            print("  (Ollama not reachable — cannot list local models)")
        print()
        return

    if args.pull:
        try:
            import ollama as _ollama

            print(f"  Pulling '{args.pull}'...")
            for chunk in _ollama.pull(args.pull, stream=True):
                status = (
                    chunk.get("status", "")
                    if isinstance(chunk, dict)
                    else getattr(chunk, "status", "")
                )
                total = (
                    chunk.get("total", 0) if isinstance(chunk, dict) else getattr(chunk, "total", 0)
                )
                completed = (
                    chunk.get("completed", 0)
                    if isinstance(chunk, dict)
                    else getattr(chunk, "completed", 0)
                )
                if total and completed:
                    pct = int(completed / total * 100)
                    print(f"\r  {status}: {pct}%  ", end="", flush=True)
                elif status:
                    print(f"\r  {status}...    ", end="", flush=True)
            print(f"\n  '{args.pull}' is ready.\n")
        except Exception as e:
            print(f"\n  Error: Failed to pull '{args.pull}': {e}")
        return

    # Auto-pull Ollama model if not available locally
    if config.llm_provider == "ollama" and config.llm_model:
        try:
            import ollama as _ollama

            response = _ollama.list()
            models = response.models if hasattr(response, "models") else response.get("models", [])
            local_names = set()
            for m in models:
                name = m.model if hasattr(m, "model") else m.get("name", "")
                local_names.add(name)
                local_names.add(name.split(":")[0])  # also match without tag
            model_tag = (
                config.llm_model if ":" in config.llm_model else f"{config.llm_model}:latest"
            )
            if model_tag not in local_names and config.llm_model not in local_names:
                print(f"  Model '{config.llm_model}' not found locally — pulling from Ollama...")
                for chunk in _ollama.pull(config.llm_model, stream=True):
                    status = (
                        chunk.get("status", "")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "status", "")
                    )
                    total = (
                        chunk.get("total", 0)
                        if isinstance(chunk, dict)
                        else getattr(chunk, "total", 0)
                    )
                    completed = (
                        chunk.get("completed", 0)
                        if isinstance(chunk, dict)
                        else getattr(chunk, "completed", 0)
                    )
                    if total and completed:
                        pct = int(completed / total * 100)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    elif status:
                        print(f"\r  {status}...", end="", flush=True)
                print(f"\n  Model '{config.llm_model}' ready.\n")
        except Exception as e:
            logger.warning(f"Could not auto-pull model '{config.llm_model}': {e}")

    # Animated init display — only when entering interactive REPL
    _entering_repl = (
        not args.query
        and not getattr(args, "ingest", None)
        and not args.list
        and not args.list_models
        and not getattr(args, "pull", None)
        and not getattr(args, "refresh", False)
        and not getattr(args, "list_stale", False)
        and not getattr(args, "graph_status", False)
        and not getattr(args, "graph_finalize", False)
        and not getattr(args, "graph_export", False)
        and not getattr(args, "delete_doc", None)
        and not getattr(args, "delete_doc_id", None)
        and not getattr(args, "store_init", None)
        and not getattr(args, "share_list", False)
        and not getattr(args, "share_generate", None)
        and not getattr(args, "share_redeem", None)
        and not getattr(args, "share_revoke", None)
        and not getattr(args, "session_list", False)
        and sys.stdin.isatty()
    )
    _init_display: _InitDisplay | None = None
    _saved_propagate: dict = {}
    _INIT_LOGGER_NAMES = [
        "Axon",
        "Axon.Retrievers",
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers",
        "chromadb",
        "chromadb.telemetry.product.posthog",
        "httpx",
    ]
    if _entering_repl:
        print()
        _init_display = _InitDisplay()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _saved_propagate[_n] = _lg.propagate
            _lg.propagate = False  # suppress default stderr handler
            _lg.setLevel(logging.INFO)
            _lg.addHandler(_init_display)

    brain = AxonBrain(config)

    if _init_display:
        _init_display.stop()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _lg.removeHandler(_init_display)
            _lg.propagate = _saved_propagate.get(_n, True)

    # --- Project CLI handling ---
    from axon.projects import (
        ProjectHasChildrenError,
        delete_project,
        ensure_project,
        list_projects,
        project_dir,
    )

    if args.project_list:
        projects = list_projects()
        if not projects:
            print("  No projects yet. Use --project-new <name> to create one.")
        else:
            print()
            active = brain._active_project

            _print_project_tree(projects, active)
            print(f"\n  Active: {active}")
        return

    if args.project_delete:
        proj_name = args.project_delete.lower()
        try:
            if brain._active_project == proj_name:
                brain.switch_project("default")
            delete_project(proj_name)
            print(f"  Deleted project '{proj_name}'.")
        except ProjectHasChildrenError as e:
            print(f"  {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"  {e}")
            sys.exit(1)
        return

    # Switch to an existing project
    if args.project:
        proj_name = args.project.lower()
        try:
            brain.switch_project(proj_name)
        except ValueError as e:
            print(f"  {e}")
            sys.exit(1)

    # Create (if needed) and switch to new project
    if args.project_new:
        proj_name = args.project_new.lower()
        ensure_project(proj_name)
        brain.switch_project(proj_name)
        print(f"  Using project '{proj_name}'  ({project_dir(proj_name)})")

    if args.ingest:
        if os.path.isdir(args.ingest):
            asyncio.run(brain.load_directory(args.ingest))
        else:
            from axon.loaders import DirectoryLoader

            ext = os.path.splitext(args.ingest)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders:
                docs = loader_mgr.loaders[ext].load(args.ingest)
                # Add [File Path:] breadcrumb to match directory ingest metadata
                abs_path = os.path.abspath(args.ingest)
                for doc in docs:
                    if doc.get("metadata", {}).get("type") not in ("csv", "tsv", "image"):
                        doc["text"] = f"[File Path: {abs_path}]\n{doc['text']}"
                brain.ingest(docs)

    if args.list:
        docs = brain.list_documents()
        if not docs:
            print("  Knowledge base is empty.")
        else:
            total_chunks = sum(d["chunks"] for d in docs)
            print(f"\n  Knowledge Base — {len(docs)} file(s), {total_chunks} chunk(s)\n")
            print(f"  {'Source':<60} {'Chunks':>6}")
            print(f"  {'-'*60} {'-'*6}")
            for d in docs:
                print(f"  {d['source']:<60} {d['chunks']:>6}")
        return

    if getattr(args, "refresh", False):
        import hashlib as _hashlib

        from axon.loaders import DirectoryLoader

        versions = brain.get_doc_versions()
        if not versions:
            print("  No tracked documents to refresh.")
            return
        loader_mgr = DirectoryLoader()
        reingested, skipped, missing, errors = [], [], [], []
        for source_id, record in versions.items():
            if not os.path.exists(source_id):
                missing.append(source_id)
                continue
            ext = os.path.splitext(source_id)[1].lower()
            loader_instance = loader_mgr.loaders.get(ext)
            if loader_instance is None:
                errors.append(f"{source_id}: no loader for extension '{ext}'")
                continue
            try:
                docs = loader_instance.load(source_id)
                if not docs:
                    errors.append(f"{source_id}: loader returned no documents")
                    continue
                combined = "".join(d.get("text", "") for d in docs)
                current_hash = _hashlib.md5(combined.encode("utf-8", errors="replace")).hexdigest()
                if current_hash == record.get("content_hash"):
                    skipped.append(source_id)
                    continue
                brain.ingest(docs)
                reingested.append(source_id)
            except Exception as exc:
                errors.append(f"{source_id}: {exc}")
        print("\n  Refresh complete")
        print(f"  Re-ingested : {len(reingested)}")
        print(f"  Unchanged   : {len(skipped)}")
        print(f"  Missing     : {len(missing)}")
        if errors:
            print(f"  Errors      : {len(errors)}")
            for e in errors:
                print(f"    • {e}")
        return

    if getattr(args, "list_stale", False):
        from datetime import datetime, timezone

        versions = brain.get_doc_versions()
        days = getattr(args, "stale_days", 7)
        cutoff = datetime.now(timezone.utc).timestamp() - days * 86_400
        stale = []
        for source_id, record in versions.items():
            try:
                ingested_ts = datetime.fromisoformat(
                    record.get("ingested_at", "").replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                continue
            if ingested_ts < cutoff:
                age_days = round((datetime.now(timezone.utc).timestamp() - ingested_ts) / 86_400, 1)
                stale.append((source_id, age_days))
        if not stale:
            print(f"  No documents older than {days} day(s).")
        else:
            print(f"\n  Stale documents (>{days} days):\n")
            for src, age in sorted(stale, key=lambda x: -x[1]):
                print(f"  {age:>6.1f}d  {src}")
        return

    if getattr(args, "graph_status", False):
        entity_count = len(getattr(brain, "_entity_graph", {}) or {})
        code_node_count = len((getattr(brain, "_code_graph", {}) or {}).get("nodes", {}))
        summary_count = len(getattr(brain, "_community_summaries", {}) or {})
        in_progress = getattr(brain, "_community_build_in_progress", False)
        graph_ready = entity_count > 0 or code_node_count > 0
        print("\n  Graph Status")
        print(f"  Ready              : {'yes' if graph_ready else 'no'}")
        print(f"  Entities           : {entity_count}")
        print(f"  Code nodes         : {code_node_count}")
        print(f"  Community summaries: {summary_count}")
        print(f"  Build in progress  : {'yes' if in_progress else 'no'}")
        return

    if getattr(args, "graph_finalize", False):
        print("  Finalizing graph (community rebuild)...")
        try:
            brain.finalize_graph(True)
            summary_count = len(getattr(brain, "_community_summaries", {}) or {})
            print(f"  Done. {summary_count} community summaries generated.")
        except Exception as exc:
            print(f"  Error: {exc}")
            sys.exit(1)
        return

    if getattr(args, "graph_export", False):
        import tempfile

        print("  Exporting graph...")
        try:
            cache_dir = os.path.join(os.path.expanduser("~"), ".axon", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            out_path = os.path.join(cache_dir, "graph.html")
        except OSError:
            out_path = os.path.join(tempfile.gettempdir(), "axon_graph.html")
        try:
            html = brain.export_graph_html(open_browser=False)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"  Graph exported to: {out_path}")
        except Exception as exc:
            print(f"  Error: {exc}")
            sys.exit(1)
        return

    if getattr(args, "delete_doc", None):
        source = args.delete_doc
        docs = brain.list_documents()
        match = [d for d in docs if d["source"] == source or source in d["source"]]
        if not match:
            print(f"  No documents matching source '{source}'.")
            sys.exit(1)
        ids_to_delete = [i for d in match for i in d.get("doc_ids", [])]
        brain.vector_store.delete_by_ids(ids_to_delete)
        if brain.bm25 is not None:
            brain.bm25.delete_documents(ids_to_delete)
        total_chunks = sum(d["chunks"] for d in match)
        print(f"  Deleted {total_chunks} chunk(s) from '{source}'.")
        return

    if getattr(args, "delete_doc_id", None):
        ids_to_delete = args.delete_doc_id
        existing = brain.vector_store.get_by_ids(ids_to_delete)
        existing_ids = [d["id"] for d in existing]
        not_found = [i for i in ids_to_delete if i not in existing_ids]
        if existing_ids:
            brain.vector_store.delete_by_ids(existing_ids)
            if brain.bm25 is not None:
                brain.bm25.delete_documents(existing_ids)
        print(f"  Deleted: {len(existing_ids)}  Not found: {len(not_found)}")
        return

    if getattr(args, "store_init", None):
        import getpass as _gp

        from axon.projects import ensure_user_namespace

        base = Path(args.store_init).expanduser().resolve()
        username = _gp.getuser()
        store_root = base / "AxonStore"
        user_dir = store_root / username
        try:
            ensure_user_namespace(user_dir)
            brain.config.axon_store_base = str(base)
            brain.config.axon_store_mode = True
            brain.config.projects_root = str(user_dir)
            brain.config.vector_store_path = str(user_dir / "default" / "chroma_data")
            brain.config.bm25_path = str(user_dir / "default" / "bm25_index")
            try:
                brain.config.save()
            except Exception:
                pass
            print(f"  AxonStore initialised at {store_root}")
            print(f"  User directory : {user_dir}")
            print(f"  Username       : {username}")
        except Exception as exc:
            print(f"  Store init failed: {exc}")
            sys.exit(1)
        return

    if getattr(args, "share_list", False):
        if not brain.config.axon_store_mode:
            print("  AxonStore mode is not active. Run --store-init <path> first.")
            sys.exit(1)
        from axon import shares as _shares_mod

        user_dir = Path(brain.config.projects_root)
        _shares_mod.validate_received_shares(user_dir)
        data = _shares_mod.list_shares(user_dir)
        sharing = data.get("sharing", [])
        shared = data.get("shared", [])
        print("\n  Shares — issued by me:")
        if sharing:
            for s in sharing:
                tag = " [revoked]" if s.get("revoked") else ""
                print(f"    {s['project']} → {s['grantee']}  [ro]{tag}")
        else:
            print("    (none)")
        print("\n  Shares — received:")
        if shared:
            for s in shared:
                print(f"    {s['owner']}/{s['project']} mounted as {s.get('mount', '')}")
        else:
            print("    (none)")
        print()
        return

    if getattr(args, "share_generate", None):
        if not brain.config.axon_store_mode:
            print("  AxonStore mode is not active. Run --store-init <path> first.")
            sys.exit(1)
        from axon import shares as _shares_mod

        proj, grantee = args.share_generate
        user_dir = Path(brain.config.projects_root)
        _segs = proj.split("/")
        proj_dir = user_dir / _segs[0]
        for _seg in _segs[1:]:
            proj_dir = proj_dir / "subs" / _seg
        if not proj_dir.exists() or not (proj_dir / "meta.json").exists():
            print(f"  Project '{proj}' not found.")
            sys.exit(1)
        try:
            result = _shares_mod.generate_share_key(
                owner_user_dir=user_dir, project=proj, grantee=grantee
            )
            print(f"\n  Share key generated for project '{proj}'")
            print(f"  Grantee:  {grantee}")
            print(f"  Key ID:   {result['key_id']}")
            print(f"\n  Share string:\n\n    {result['share_string']}\n")
            print(f"  Revoke:   axon --share-revoke {result['key_id']}\n")
        except Exception as exc:
            print(f"  Share generation failed: {exc}")
            sys.exit(1)
        return

    if getattr(args, "share_redeem", None):
        if not brain.config.axon_store_mode:
            print("  AxonStore mode is not active. Run --store-init <path> first.")
            sys.exit(1)
        from axon import shares as _shares_mod

        user_dir = Path(brain.config.projects_root)
        try:
            result = _shares_mod.redeem_share_key(
                grantee_user_dir=user_dir, share_string=args.share_redeem.strip()
            )
            print("\n  Share redeemed!")
            print(f"  Project '{result['project']}' from {result['owner']}")
            mount = result.get("mount_name", result["owner"] + "_" + result["project"])
            print(f"  Mounted at:  mounts/{mount}")
            print("  Access:      read-only\n")
        except Exception as exc:
            print(f"  Redeem failed: {exc}")
            sys.exit(1)
        return

    if getattr(args, "share_revoke", None):
        if not brain.config.axon_store_mode:
            print("  AxonStore mode is not active. Run --store-init <path> first.")
            sys.exit(1)
        from axon import shares as _shares_mod

        user_dir = Path(brain.config.projects_root)
        try:
            result = _shares_mod.revoke_share_key(
                owner_user_dir=user_dir, key_id=args.share_revoke.strip()
            )
            print(f"  Share '{result['key_id']}' revoked.")
        except Exception as exc:
            print(f"  Revoke failed: {exc}")
            sys.exit(1)
        return

    if getattr(args, "session_list", False):
        from axon.sessions import _list_sessions, _print_sessions

        sessions = _list_sessions(project=brain._active_project)
        if not sessions:
            print("  No saved sessions.")
        else:
            _print_sessions(sessions)
        return

    if args.query:
        if getattr(args, "dry_run", False):
            results, diag, trace = brain.search_raw(args.query)
            import json as _json_cli

            print(f"\n  [DRY RUN] {diag.result_count} chunk(s) retrieved")
            print(f"  Diagnostics:\n{_json_cli.dumps(diag.to_dict(), indent=4)}")
            print("\n  Ranked chunks:")
            for i, r in enumerate(results, 1):
                meta = r.get("metadata", {})
                src = meta.get("source") or meta.get("file_path") or r["id"]
                sym = meta.get("symbol_name", "")
                label = f"{src} :: {sym}" if sym else src
                print(f"  {i:>2}. [{r['score']:.3f}]  {label}")
                print(f"      {r['text'][:100]!r}")
        elif args.stream:
            for chunk in brain.query_stream(args.query):
                if isinstance(chunk, dict):
                    continue
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"\n  Response:\n{brain.query(args.query)}")
        return

    # No query supplied — enter interactive REPL (streaming on by default)
    _quiet = args.quiet or not sys.stdin.isatty()
    try:
        _interactive_repl(brain, stream=True, init_display=_init_display, quiet=_quiet)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n  Bye!")
    # Manually flush readline history, then hard-exit to skip atexit handlers
    # (colorama/posthog atexit callbacks raise tracebacks on double Ctrl+C)
    try:
        import readline as _rl

        _hist = os.path.expanduser("~/.axon_history")
        _rl.write_history_file(_hist)
    except Exception:
        pass
    os._exit(0)
