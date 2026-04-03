"""Interactive configuration wizard for Axon.


Provides:


  - run_wizard()         — interactive config editor (numbered menus + validated inputs)


  - render_issues()      — rich-formatted validation report


  - render_config_table() — rich table of current config grouped by section


"""


from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def render_issues(issues: list) -> None:
    """Print a rich-formatted validation report to the terminal.

    *issues* is a list of :class:`axon.config.ConfigIssue` instances.

    Falls back to plain-text output when *rich* is unavailable.

    """

    if not issues:
        try:
            from rich.console import Console

            Console().print("[bold green]No issues found. Config looks good![/bold green]")

        except ImportError:
            print("No issues found. Config looks good!")

        return

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(
            title="Config Validation Report",
            box=box.ROUNDED,
            show_lines=True,
            expand=False,
        )

        table.add_column("Level", style="bold", width=7)

        table.add_column("Section", width=14)

        table.add_column("Field", width=30)

        table.add_column("Message")

        table.add_column("Suggestion")

        _LEVEL_STYLES = {
            "error": "bold red",
            "warn": "bold yellow",
            "info": "bold green",
        }

        for issue in issues:
            style = _LEVEL_STYLES.get(issue.level, "")

            table.add_row(
                f"[{style}]{issue.level.upper()}[/{style}]",
                issue.section,
                issue.field,
                issue.message,
                issue.suggestion or "",
            )

        console.print(table)

    except ImportError:
        # Plain-text fallback

        for issue in issues:
            line = f"[{issue.level.upper()}] {issue.section}.{issue.field}: {issue.message}"

            if issue.suggestion:
                line += f" — {issue.suggestion}"

            print(line)


def render_config_table(config: Any) -> None:
    """Print the current config as a rich table grouped by section.

    *config* is an :class:`axon.config.AxonConfig` instance.

    Falls back to plain-text key=value output when *rich* is unavailable.

    """

    # Define the sections to display (field_name, display_label)

    _SECTIONS: dict[str, list[tuple[str, str]]] = {
        "LLM": [
            ("llm_provider", "provider"),
            ("llm_model", "model"),
            ("ollama_base_url", "base_url"),
            ("llm_temperature", "temperature"),
            ("llm_max_tokens", "max_tokens"),
        ],
        "Embedding": [
            ("embedding_provider", "provider"),
            ("embedding_model", "model"),
        ],
        "Vector Store": [
            ("vector_store", "provider"),
            ("vector_store_path", "path"),
        ],
        "Chunk": [
            ("chunk_strategy", "strategy"),
            ("chunk_size", "size"),
            ("chunk_overlap", "overlap"),
        ],
        "RAG": [
            ("top_k", "top_k"),
            ("similarity_threshold", "similarity_threshold"),
            ("hybrid_search", "hybrid_search"),
            ("rerank", "rerank"),
            ("sentence_window", "sentence_window"),
            ("sentence_window_size", "sentence_window_size"),
            ("graph_rag", "graph_rag"),
            ("graph_rag_mode", "graph_rag_mode"),
            ("raptor", "raptor"),
            ("query_router", "query_router"),
        ],
        "Store": [
            ("axon_store_base", "base"),
            ("projects_root", "projects_root"),
        ],
        "Offline": [
            ("offline_mode", "enabled"),
            ("local_assets_only", "local_assets_only"),
        ],
    }

    try:
        from rich import box
        from rich.console import Console
        from rich.table import Table

        console = Console()

        for section_name, fields in _SECTIONS.items():
            table = Table(
                title=f"[bold]{section_name}[/bold]",
                box=box.SIMPLE,
                show_header=True,
                expand=False,
                min_width=50,
            )

            table.add_column("Field", style="cyan", width=30)

            table.add_column("Value", style="white")

            for attr, label in fields:
                value = getattr(config, attr, "N/A")

                table.add_row(label, str(value))

            console.print(table)

    except ImportError:
        # Plain-text fallback

        for section_name, fields in _SECTIONS.items():
            print(f"\n  [{section_name}]")

            for attr, label in fields:
                value = getattr(config, attr, "N/A")

                print(f"    {label} = {value}")

        print()


def run_wizard(brain: Any = None, config_path: str = "") -> dict[str, Any]:
    """Run the interactive config wizard in the terminal.

    Starts by asking the user to choose a setup mode:

    * **Quick**    — only LLM, Embedding, Vector Store (get running in 2 minutes).

    * **Standard** — common settings: all of Quick plus Chunking, Retrieval,

      Reranking, basic Query Transformations, Output, Offline, and Store.

    * **Full**     — every AxonConfig parameter across all 16 sections.

    Returns a dict of changed fields.  The caller is responsible for saving.

    If *brain* is provided its ``brain.config`` is used to pre-fill values.

    Choice fields display a numbered menu — type the number or the value.

    Numeric fields validate the input on the spot and re-ask on bad input.

    API key fields show '***set***' when already configured.

    Answer 'n' at any section prompt to skip that section.

    """

    try:
        from rich.console import Console

        _console = Console()

        _header = lambda msg: _console.print(f"\n[bold cyan]{msg}[/bold cyan]")  # noqa: E731

        _label = lambda msg: _console.print(f"  [dim]{msg}[/dim]")  # noqa: E731

        _warn = lambda msg: _console.print(f"  [bold yellow]⚠  {msg}[/bold yellow]")  # noqa: E731

        _err = lambda msg: _console.print(f"  [red]{msg}[/red]")  # noqa: E731

    except ImportError:
        _header = print

        _label = print

        _warn = lambda msg: print(f"  WARNING: {msg}")  # noqa: E731

        _err = print

    cfg = brain.config if (brain is not None and hasattr(brain, "config")) else None

    def _g(field: str, default: Any) -> Any:
        """Get field value from live config, falling back to *default*."""

        return getattr(cfg, field, default) if cfg else default

    # ── Core input helpers ────────────────────────────────────────────────────

    def _pick(label: str, field: str, choices: list[str], default: Any) -> None:
        """Display a numbered menu and loop until the user picks a valid item.

        Accepts either the number (1-N) or the value string itself.

        For boolean choices the value stored in *changes* is a Python bool.

        """

        raw = _g(field, default)

        cur_str = str(raw).lower() if isinstance(raw, bool) else str(raw)

        # Find which menu item matches the current value

        cur_idx = next((i for i, c in enumerate(choices) if c.lower() == cur_str.lower()), 0)

        print(f"\n  {label}:")

        for i, c in enumerate(choices, 1):
            marker = "  ◀ current" if i - 1 == cur_idx else ""

            print(f"    {i}) {c}{marker}")

        while True:
            try:
                ans = input(
                    f"  Select [1–{len(choices)}, Enter = keep '{choices[cur_idx]}']: "
                ).strip()

            except EOFError:
                print()

                return

            except KeyboardInterrupt:
                print()

                raise

            if not ans:
                return  # keep current

            # Try as index number

            try:
                idx = int(ans)

                if 1 <= idx <= len(choices):
                    selected = choices[idx - 1]

                    break

                _err(f"  ✗ Enter a number between 1 and {len(choices)}.")

                continue

            except ValueError:
                pass

            # Try as the literal value

            match = next((c for c in choices if c.lower() == ans.lower()), None)

            if match:
                selected = match

                break

            _err(
                f"  ✗ '{ans}' is not a valid option. "
                f"Enter a number (1–{len(choices)}) or the value directly."
            )

        # Store as Python bool for boolean-style menus

        _bool_choices = {"true", "false"}

        if {c.lower() for c in choices} == _bool_choices:
            val: Any = selected.lower() in ("true", "1", "yes")

        else:
            val = selected

        if val != _g(field, default):
            changes[field] = val

    def _ask_int(
        label: str,
        field: str,
        default: int,
        min_val: int | None = None,
        max_val: int | None = None,
    ) -> None:
        """Prompt for an integer, loop until the value is valid."""

        cur = _g(field, default)

        parts = [f"current: {cur}"]

        if min_val is not None and max_val is not None:
            parts.append(f"range {min_val}–{max_val}")

        elif min_val is not None:
            parts.append(f"min {min_val}")

        elif max_val is not None:
            parts.append(f"max {max_val}")

        hint = ", ".join(parts)

        while True:
            try:
                ans = input(f"  {label} [{hint}]: ").strip()

            except EOFError:
                print()

                return

            except KeyboardInterrupt:
                print()

                raise

            if not ans:
                return  # keep current

            try:
                val = int(ans)

            except ValueError:
                _err(f"  ✗ '{ans}' is not a valid integer. Try again or press Enter to keep {cur}.")

                continue

            if min_val is not None and val < min_val:
                _err(f"  ✗ Value must be ≥ {min_val}.")

                continue

            if max_val is not None and val > max_val:
                _err(f"  ✗ Value must be ≤ {max_val}.")

                continue

            if val != cur:
                changes[field] = val

            return

    def _ask_float(
        label: str,
        field: str,
        default: float,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        """Prompt for a float, loop until the value is valid."""

        cur = _g(field, default)

        parts = [f"current: {cur}"]

        if min_val is not None and max_val is not None:
            parts.append(f"range {min_val}–{max_val}")

        elif min_val is not None:
            parts.append(f"min {min_val}")

        elif max_val is not None:
            parts.append(f"max {max_val}")

        hint = ", ".join(parts)

        while True:
            try:
                ans = input(f"  {label} [{hint}]: ").strip()

            except EOFError:
                print()

                return

            except KeyboardInterrupt:
                print()

                raise

            if not ans:
                return

            try:
                val = float(ans)

            except ValueError:
                _err(f"  ✗ '{ans}' is not a valid number. Try again or press Enter to keep {cur}.")

                continue

            if min_val is not None and val < min_val:
                _err(f"  ✗ Value must be ≥ {min_val}.")

                continue

            if max_val is not None and val > max_val:
                _err(f"  ✗ Value must be ≤ {max_val}.")

                continue

            if val != cur:
                changes[field] = val

            return

    def _ask_str(label: str, field: str, default: str) -> None:
        """Prompt for a free-text string value."""

        cur = str(_g(field, default))

        try:
            ans = input(f"  {label} [current: {cur!r}]: ").strip()

        except EOFError:
            print()

            return

        except KeyboardInterrupt:
            print()

            raise

        if ans and ans != cur:
            changes[field] = ans

    def _ask_secret(label: str, field: str) -> None:
        """Prompt for a sensitive value; shows '***set***' when already set."""

        cur = str(_g(field, ""))

        display = "***set***" if cur else "(not set)"

        try:
            ans = input(f"  {label} [current: {display}] (Enter = keep): ").strip()

        except EOFError:
            print()

            return

        except KeyboardInterrupt:
            print()

            raise

        if ans and ans != "***set***":
            changes[field] = ans

    def _section_skip(name: str) -> bool:
        """Prompt user whether to enter a section; True = skip."""

        try:
            ans = input(f"\n  Configure '{name}' section? [Y/n]: ").strip().lower()

        except EOFError:
            return False

        except KeyboardInterrupt:
            raise

        return ans in ("n", "no", "skip")

    # ── Shared choice lists ───────────────────────────────────────────────────

    _YN = ["true", "false"]

    changes: dict[str, Any] = {}

    # ── Mode selection ────────────────────────────────────────────────────────

    _header("Axon Config Wizard")

    _label("Choose how much of the configuration you want to set up:")

    _MODE_CHOICES = ["quick", "standard", "full"]

    _MODE_DESC = {
        "quick": "Essential settings only — LLM, Embedding, Vector Store (2 min).",
        "standard": "Common settings — adds Chunking, Retrieval, Reranking, Query "
        "Transformations, Output, and Offline (10 min).",
        "full": "Every parameter — all 16 sections including RAPTOR, GraphRAG, "
        "Code Graph, and advanced tuning knobs (expert).",
    }

    print()

    for i, m in enumerate(_MODE_CHOICES, 1):
        print(f"    {i}) {m:10s}  {_MODE_DESC[m]}")

    _mode = "standard"  # safe default

    while True:
        try:
            ans = (
                input(f"\n  Setup mode [1–{len(_MODE_CHOICES)}, Enter = standard]: ")
                .strip()
                .lower()
            )

        except EOFError:
            print()

            break

        except KeyboardInterrupt:
            print("\n  Setup cancelled.")

            return {}

        if not ans:
            break

        try:
            idx = int(ans)

            if 1 <= idx <= len(_MODE_CHOICES):
                _mode = _MODE_CHOICES[idx - 1]

                break

            _err(f"  ✗ Enter a number between 1 and {len(_MODE_CHOICES)}.")

        except ValueError:
            if ans in _MODE_CHOICES:
                _mode = ans

                break

            _err(
                f"  ✗ '{ans}' is not a valid mode. Enter a number or one of: {', '.join(_MODE_CHOICES)}."
            )

    _header(f"Mode: {_mode.upper()} — {_MODE_DESC[_mode]}")

    _label("Numbered menus: type the number or the value, then Enter.")

    _label("Free-text fields: type the new value, or press Enter to keep current.")

    _label("Answer 'n' at any section prompt to skip that section.")

    _label("Press Ctrl+C at any prompt to exit immediately without saving.")

    def _in(*modes: str) -> bool:
        """Return True if the current setup mode is one of *modes*."""

        return _mode in modes

    # ── 1. LLM  [quick / standard / full] ────────────────────────────────────

    _header("1 / 16  LLM")

    _label("Language model for answer generation and advanced strategies (HyDE, RAPTOR, GraphRAG).")

    _label("ollama runs locally; openai/gemini/grok/vllm require credentials.")

    if not _section_skip("LLM"):
        _pick(
            "llm.provider — which LLM backend to use",
            "llm_provider",
            ["ollama", "openai", "gemini", "grok", "vllm", "github_copilot"],
            "ollama",
        )

        _label("Model name for the chosen provider (e.g. llama3.1:8b, gpt-4o, gemini-2.0-flash).")

        _ask_str("llm.model", "llm_model", "llama3.1:8b")

        _label("Ollama / vLLM server base URL (ignored for cloud providers).")

        _ask_str("llm.base_url", "ollama_base_url", "http://localhost:11434")

        _label("API keys — only the key for your chosen provider is needed.")

        _ask_secret("llm.openai_api_key  (openai)", "openai_api_key")

        _ask_secret("llm.grok_api_key    (grok)", "grok_api_key")

        _ask_secret("llm.gemini_api_key  (gemini)", "gemini_api_key")

        _ask_secret("llm.copilot_pat     (github_copilot)", "copilot_pat")

        if _in("standard", "full"):
            _label("temperature: 0.0 = deterministic, 1.0 = creative. Recommended 0.3–0.8.")

            _ask_float("llm.temperature", "llm_temperature", 0.7, 0.0, 2.0)

            _label("max_tokens: maximum tokens in the LLM response. Recommended 1024–8192.")

            _ask_int("llm.max_tokens", "llm_max_tokens", 2048, 1, 65536)

        if _in("full"):
            _label("timeout: seconds before giving up on an LLM call. Recommended 30–120.")

            _ask_int("llm.timeout", "llm_timeout", 60, 1, 600)

            _label("vllm_base_url: only used when provider=vllm.")

            _ask_str("llm.vllm_base_url", "vllm_base_url", "http://localhost:8000/v1")

    # ── 2. Embedding  [quick / standard / full] ───────────────────────────────

    _header("2 / 16  Embedding")

    _label(
        "Converts text to vectors for semantic search. Changing the model invalidates existing indexes."
    )

    if not _section_skip("Embedding"):
        _pick(
            "embedding.provider",
            "embedding_provider",
            ["sentence_transformers", "ollama", "fastembed", "openai"],
            "sentence_transformers",
        )

        _label(
            "Model name (e.g. all-MiniLM-L6-v2, BAAI/bge-large-en-v1.5, text-embedding-3-small)."
        )

        _ask_str("embedding.model", "embedding_model", "all-MiniLM-L6-v2")

        # model_path: standard+

        _label(
            "embedding.model_path: absolute path to a local model folder (skips download). Leave blank to auto-download."
        )

        _ask_str("embedding.model_path", "embedding_model_path", "")

    # ── 3. Vector Store  [quick / standard / full] ────────────────────────────

    _header("3 / 16  Vector Store")

    _label(
        "turboquantdb (tqdb) = default — fastest ingest, smallest disk, no service needed."
        "  lancedb = local columnar store.  qdrant = local or remote server.  chroma = local."
    )

    if not _section_skip("Vector Store"):
        _pick(
            "vector_store.provider",
            "vector_store",
            ["turboquantdb", "lancedb", "qdrant", "chroma"],
            "turboquantdb",
        )

        # path + qdrant creds: standard+

        _label("vector_store.path: storage directory (blank = use default project path).")

        _ask_str("vector_store.path", "vector_store_path", "")

        _label("qdrant fields are only used when provider=qdrant.")

        _ask_str("qdrant.url", "qdrant_url", "")

        _ask_secret("qdrant.api_key (Qdrant Cloud)", "qdrant_api_key")

        # tqdb fields: full mode only
        if _in("full"):
            _label(
                "turboquantdb fields are only used when provider=turboquantdb."
                "  bits=8 + rerank=true is the recommended preset (perfect recall, ~2× compression)."
            )
            _pick("vector_store.tqdb_bits", "tqdb_bits", [4, 8], 8)
            _pick("vector_store.tqdb_fast_mode", "tqdb_fast_mode", [False, True], False)
            _pick("vector_store.tqdb_rerank", "tqdb_rerank", [True, False], True)

    # ── 4. Chunking  [standard / full] ────────────────────────────────────────

    if _in("standard", "full"):
        _header("4 / 16  Chunking")

        _label(
            "How documents are split at ingest time. Changes only affect newly ingested documents."
        )

        _label("  recursive       — paragraph/sentence/word boundaries, fast, general-purpose")

        _label("  semantic        — group by embedding similarity, default, best quality")

        _label("  markdown        — split on Markdown headers, ideal for docs/wikis")

        _label("  cosine_semantic — cosine similarity grouping, slowest, finest-grained")

        if not _section_skip("Chunking"):
            _pick(
                "chunk.strategy",
                "chunk_strategy",
                ["recursive", "semantic", "markdown", "cosine_semantic"],
                "semantic",
            )

            _label(
                "chunk.size: target tokens per chunk. Smaller = more precise; larger = more context."
            )

            _ask_int("chunk.size", "chunk_size", 1000, 50, 10000)

            _label(
                "chunk.overlap: token overlap between consecutive chunks to preserve cross-boundary context."
            )

            _ask_int("chunk.overlap", "chunk_overlap", 200, 0, 5000)

            # cosine params: standard+

            _label(
                "cosine_semantic_threshold: merge distance threshold (0.0–1.0). Only active for cosine_semantic strategy."
            )

            _ask_float(
                "chunk.cosine_semantic_threshold", "cosine_semantic_threshold", 0.7, 0.0, 1.0
            )

            _label("cosine_semantic_max_size: max tokens per merged group (cosine_semantic only).")

            _ask_int("chunk.cosine_semantic_max_size", "cosine_semantic_max_size", 500, 10, 5000)

    # ── 5. Retrieval  [standard / full] ───────────────────────────────────────

    if _in("standard", "full"):
        _header("5 / 16  Retrieval")

        _label("Core retrieval knobs — the most impactful settings for search quality.")

        if not _section_skip("Retrieval"):
            _label(
                "top_k: chunks retrieved per query. Higher = richer context but more tokens and latency."
            )

            _ask_int("rag.top_k", "top_k", 10, 1, 200)

            _label("similarity_threshold: minimum cosine similarity (0.0–1.0) to include a chunk.")

            _ask_float("rag.similarity_threshold", "similarity_threshold", 0.3, 0.0, 1.0)

            _label(
                "hybrid_search: combine vector search with BM25 keyword search for better recall."
            )

            _pick("rag.hybrid_search", "hybrid_search", _YN, True)

            # hybrid tuning + mmr + parent: standard+

            _label(
                "hybrid_weight: semantic weight when hybrid_mode=weighted. 1.0 = pure semantic, 0.0 = pure keyword."
            )

            _ask_float("rag.hybrid_weight", "hybrid_weight", 0.7, 0.0, 1.0)

            _pick(
                "rag.hybrid_mode — rrf is more robust; weighted lets you tune the balance above",
                "hybrid_mode",
                ["rrf", "weighted"],
                "rrf",
            )

            _label(
                "mmr: Maximal Marginal Relevance — reorder results to reduce near-duplicate chunks."
            )

            _pick("rag.mmr", "mmr", _YN, False)

            _label(
                "mmr_lambda: 1.0 = pure relevance, 0.0 = pure diversity. Only active when mmr=true."
            )

            _ask_float("rag.mmr_lambda", "mmr_lambda", 0.5, 0.0, 1.0)

            _label(
                "parent_chunk_size: index small child chunks but return this larger parent to the LLM. 0 = disabled."
            )

            _ask_int("rag.parent_chunk_size", "parent_chunk_size", 1500, 0, 50000)

    # ── 6. Sentence Window & CRAG-Lite  [standard / full] ────────────────────

    if _in("standard", "full"):
        _header("6 / 16  Sentence Window & CRAG-Lite")

        if not _section_skip("Sentence Window & CRAG-Lite"):
            _label(
                "sentence_window: index at sentence granularity but expand hits with surrounding sentences."
            )

            _pick("rag.sentence_window", "sentence_window", _YN, False)

            _label(
                "sentence_window_size: sentences of context on each side of the matched sentence."
            )

            _ask_int("rag.sentence_window_size", "sentence_window_size", 3, 1, 20)

            # crag_lite: standard+

            _label(
                "crag_lite: Corrective RAG-Lite — scores retrieval confidence; falls back when low. No extra LLM calls."
            )

            _pick("rag.crag_lite", "crag_lite", _YN, False)

            _label(
                "crag_lite_confidence_threshold: below this score the query falls back. 0.0–1.0."
            )

            _ask_float(
                "rag.crag_lite_confidence_threshold",
                "crag_lite_confidence_threshold",
                0.4,
                0.0,
                1.0,
            )

    # ── 7. Reranking  [standard / full] ───────────────────────────────────────

    if _in("standard", "full"):
        _header("7 / 16  Reranking")

        _label(
            "Re-scores retrieved chunks for precision. cross-encoder uses a small local model (no LLM calls)."
        )

        if not _section_skip("Reranking"):
            _pick("rerank.enabled", "rerank", _YN, False)

            _pick(
                "rerank.provider — cross-encoder: fast local model  |  llm: uses your LLM (adds latency)",
                "reranker_provider",
                ["cross-encoder", "llm"],
                "cross-encoder",
            )

            # model name: standard+

            _label("reranker_model: HuggingFace model name. BAAI/bge-reranker-v2-m3 for SOTA.")

            _ask_str("rerank.model", "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ── 8. Query Transformations  [standard / full] ───────────────────────────

    if _in("standard", "full"):
        _header("8 / 16  Query Transformations")

        _label("Each toggle adds 1 extra LLM call per query. Enable selectively.")

        if not _section_skip("Query Transformations"):
            _label(
                "multi_query: generate N paraphrased queries and merge results — improves recall."
            )

            _pick("query_transformations.multi_query", "multi_query", _YN, False)

            _label(
                "hyde: Hypothetical Document Embedding — generates a fake answer to improve retrieval."
            )

            _pick("query_transformations.hyde", "hyde", _YN, False)

            _label(
                "discussion_fallback: fall back to general LLM answer when retrieval confidence is low."
            )

            _pick("query_transformations.discussion_fallback", "discussion_fallback", _YN, True)

            # advanced transforms: standard+

            _label(
                "step_back: rephrase to a more abstract form before retrieving — helps reasoning queries."
            )

            _pick("query_transformations.step_back", "step_back", _YN, False)

            _label(
                "query_decompose: split multi-part queries into sub-queries and aggregate results."
            )

            _pick("query_transformations.query_decompose", "query_decompose", _YN, False)

            _label(
                "compress_context: remove irrelevant sentences before sending to LLM (reduces tokens)."
            )

            _pick("context_compression.enabled", "compress_context", _YN, False)

            _label(
                "  sentence = LLM-based extraction  |  llmlingua = token-level (pip install axon[llmlingua])  |  none = off"
            )

            _pick(
                "context_compression.strategy",
                "compression_strategy",
                ["sentence", "llmlingua", "none"],
                "sentence",
            )

            _pick(
                "rag.query_router — how query intent is classified",
                "query_router",
                ["heuristic", "llm", "off"],
                "heuristic",
            )

            _label(
                "contextual_retrieval: prepend chunk-level context summaries (Anthropic-style). Extra LLM calls at ingest."
            )

            _pick("rag.contextual_retrieval", "contextual_retrieval", _YN, False)

    # ── 9. RAPTOR  [standard / full] ─────────────────────────────────────────

    if _in("standard", "full"):
        _header("9 / 16  RAPTOR")

        _warn("RAPTOR builds a hierarchical summary tree using many LLM calls during ingest.")

        _warn("Best for large corpora (>5 MB). Avoid for small document sets.")

        if not _section_skip("RAPTOR"):
            _pick("rag.raptor — enable RAPTOR hierarchical indexing", "raptor", _YN, False)

            _label(
                "raptor_min_source_size_mb: skip RAPTOR for sources smaller than this (0 = no filter)."
            )

            _ask_float(
                "rag.raptor_min_source_size_mb", "raptor_min_source_size_mb", 5.0, 0.0, 1000.0
            )

            # detailed RAPTOR tuning: full only

            if _in("full"):
                _label("raptor_chunk_group_size: leaf chunks grouped per summary node.")

                _ask_int("rag.raptor_chunk_group_size", "raptor_chunk_group_size", 5, 2, 50)

                _label(
                    "raptor_max_levels: recursive summarisation depth. 1 = flat; 2–3 = multi-level."
                )

                _ask_int("rag.raptor_max_levels", "raptor_max_levels", 2, 1, 5)

                _label("raptor_cache_summaries: skip LLM calls when chunk content hasn't changed.")

                _pick("rag.raptor_cache_summaries", "raptor_cache_summaries", _YN, True)

                _label(
                    "  tree_traversal = standard  |  summary_first = prefer high-level  |  corpus_overview = global context only"
                )

                _pick(
                    "rag.raptor_retrieval_mode",
                    "raptor_retrieval_mode",
                    ["tree_traversal", "summary_first", "corpus_overview"],
                    "tree_traversal",
                )

                _label(
                    "raptor_drilldown: replace summary hits with source leaf chunks for the final answer."
                )

                _pick("rag.raptor_drilldown", "raptor_drilldown", _YN, True)

                _label("raptor_drilldown_top_k: max leaf chunks substituted per summary hit.")

                _ask_int("rag.raptor_drilldown_top_k", "raptor_drilldown_top_k", 5, 1, 50)

    # ── 10. GraphRAG  [standard / full] ───────────────────────────────────────

    if _in("standard", "full"):
        _header("10 / 16  GraphRAG")

        _warn("GraphRAG runs many LLM calls during ingest (entity extraction + relationship graph)")

        _warn("AND at query time (graph traversal + community summaries).")

        _warn("Enable only when entity-relationship reasoning is essential.")

        if not _section_skip("GraphRAG"):
            _pick(
                "rag.graph_rag — enable GraphRAG entity-centric retrieval", "graph_rag", _YN, False
            )

            _label(
                "  local = entity-centric, fast  |  global = community summaries, broad  |  hybrid = both"
            )

            _pick("rag.graph_rag_mode", "graph_rag_mode", ["local", "global", "hybrid"], "local")

            _label(
                "  light = entity list only  |  standard = + relations  |  deep = + community summaries (most LLM calls)"
            )

            _pick(
                "rag.graph_rag_depth", "graph_rag_depth", ["light", "standard", "deep"], "standard"
            )

            # detailed GraphRAG tuning: full only

            if _in("full"):
                _label(
                    "graph_rag_budget: max entity hops per query. Higher = richer context but slower."
                )

                _ask_int("rag.graph_rag_budget", "graph_rag_budget", 3, 1, 20)

                _label(
                    "graph_rag_relations: also extract and index relationships between entities."
                )

                _pick("rag.graph_rag_relations", "graph_rag_relations", _YN, True)

                _warn("Community summaries require additional LLM calls during ingest.")

                _pick("rag.graph_rag_community", "graph_rag_community", _YN, True)

                _label(
                    "graph_rag_community_async: build community summaries in background after ingest."
                )

                _pick("rag.graph_rag_community_async", "graph_rag_community_async", _YN, True)

                _label(
                    "  llm = uses your configured LLM  |  gliner = fast local NER (pip install axon[gliner])"
                )

                _pick(
                    "rag.graph_rag_ner_backend", "graph_rag_ner_backend", ["llm", "gliner"], "llm"
                )

                _label(
                    "graph_rag_entity_min_frequency: skip entities appearing fewer than N times (noise filter)."
                )

                _ask_int(
                    "rag.graph_rag_entity_min_frequency",
                    "graph_rag_entity_min_frequency",
                    2,
                    1,
                    100,
                )

                _pick(
                    "rag.graph_rag_auto_route — when to use GraphRAG vs plain retrieval",
                    "graph_rag_auto_route",
                    ["off", "heuristic", "llm"],
                    "off",
                )

                _label(
                    "graph_rag_canonicalize: merge near-duplicate entity names using LLM (extra ingest calls)."
                )

                _pick("rag.graph_rag_canonicalize", "graph_rag_canonicalize", _YN, False)

    # ── 11. Code Graph  [standard / full] ─────────────────────────────────────

    if _in("standard", "full"):
        _header("11 / 16  Code Graph")

        _label("Structural code-aware retrieval. Useful when ingesting source code repositories.")

        if not _section_skip("Code Graph"):
            _pick(
                "rag.code_graph — build symbol/call-graph from code files", "code_graph", _YN, False
            )

            _label("code_lexical_boost: identifier-aware re-scoring for code result sets.")

            _pick("rag.code_lexical_boost", "code_lexical_boost", _YN, True)

            _label("code_top_k: top-K override for code queries. 0 = use rag.top_k.")

            _ask_int("rag.code_top_k", "code_top_k", 6, 0, 100)

            # detailed code tuning: full only

            if _in("full"):
                _label(
                    "code_graph_bridge: link code symbols to prose chunks (docstrings → usage examples)."
                )

                _pick("rag.code_graph_bridge", "code_graph_bridge", _YN, False)

                _label(
                    "code_top_k_multiplier: extra fetch_k factor for diversity before final code top_k."
                )

                _ask_int("rag.code_top_k_multiplier", "code_top_k_multiplier", 2, 1, 10)

    # ── 12. Output  [standard / full] ─────────────────────────────────────────

    if _in("standard", "full"):
        _header("12 / 16  Output")

        if not _section_skip("Output"):
            _label("cite: include [Document N] inline citations in answers.")

            _pick("rag.cite", "cite", _YN, True)

            _label(
                "truth_grounding: validate answers against live web sources via Brave Search. Requires brave_api_key."
            )

            _pick("web_search.enabled", "truth_grounding", _YN, False)

            _ask_secret("web_search.brave_api_key", "brave_api_key")

    # ── 13. Performance  [standard / full] ────────────────────────────────────

    if _in("standard", "full"):
        _header("13 / 16  Performance")

        if not _section_skip("Performance"):
            _label("dedup_on_ingest: skip re-ingesting files whose content hash hasn't changed.")

            _pick("rag.dedup_on_ingest", "dedup_on_ingest", _YN, True)

            _label(
                "query_cache: cache identical query results in memory to avoid repeated LLM calls."
            )

            _pick("rag.query_cache", "query_cache", _YN, False)

            _label("query_cache_size: max cached results (LRU eviction when full).")

            _ask_int("rag.query_cache_size", "query_cache_size", 128, 8, 4096)

            _label(
                "smart_ingest: auto-detect dataset type and choose the optimal chunking strategy."
            )

            _pick("rag.smart_ingest", "smart_ingest", _YN, False)

            _label("max_workers: thread-pool size for parallel ingest and embedding.")

            _ask_int("max_workers", "max_workers", 8, 1, 64)

            # low-level ingest control: full only

            if _in("full"):
                _label(
                    "ingest_batch_mode: process ingest in batches (lower peak memory for large document sets)."
                )

                _pick("rag.ingest_batch_mode", "ingest_batch_mode", _YN, False)

                _label("max_chunks_per_source: cap chunks per source file. 0 = no limit.")

                _ask_int("rag.max_chunks_per_source", "max_chunks_per_source", 0, 0, 10000)

    # ── 14. REPL  [standard / full] ───────────────────────────────────────────

    if _in("standard", "full"):
        _header("14 / 16  REPL")

        if not _section_skip("REPL"):
            _label(
                "  local_only = allow only non-network shell commands  |  always = allow all  |  off = disallow all"
            )

            _pick(
                "repl.shell_passthrough",
                "repl_shell_passthrough",
                ["local_only", "always", "off"],
                "local_only",
            )

    # ── 15. Offline  [standard / full] ────────────────────────────────────────

    if _in("standard", "full"):
        _header("15 / 16  Offline / Air-gapped")

        _label("Prevents outbound network calls — for air-gapped environments.")

        if not _section_skip("Offline"):
            _label(
                "offline.enabled: block ALL outbound calls (LLM, embedding download, web search)."
            )

            _pick("offline.enabled", "offline_mode", _YN, False)

            _label("offline.local_assets_only: disallow URL ingest; accept only local paths.")

            _pick("offline.local_assets_only", "local_assets_only", _YN, False)

            # model dir paths: standard+

            _label(
                "Per-type model root directories. Each overrides local_models_dir for its model class."
            )

            _ask_str("offline.local_models_dir (legacy single root)", "local_models_dir", "")

            _ask_str(
                "offline.embedding_models_dir (sentence-transformers / fastembed)",
                "embedding_models_dir",
                "",
            )

            # less-common dirs: full only

            if _in("full"):
                _ask_str(
                    "offline.hf_models_dir (GLiNER, REBEL, LLMLingua, cross-encoder reranker)",
                    "hf_models_dir",
                    "",
                )

                _ask_str(
                    "offline.tokenizer_cache_dir (tiktoken BPE cache → TIKTOKEN_CACHE_DIR)",
                    "tokenizer_cache_dir",
                    "",
                )

    # ── 16. Store & Paths  [standard / full] ──────────────────────────────────

    if _in("standard", "full"):
        _header("16 / 16  Store & Paths")

        _label(
            "AxonStore is the multi-user project-sharing backend. Leave blank unless you have a shared store."
        )

        if not _section_skip("Store & Paths"):
            _ask_str(
                "store.base (root for AxonStore shared data; blank = per-user project roots)",
                "axon_store_base",
                "",
            )

            _ask_str(
                "projects_root (root for per-user projects; blank = default ~/.axon/projects)",
                "projects_root",
                "",
            )

    # ── Confirm ───────────────────────────────────────────────────────────────

    if changes:
        _header("Summary of changes")

        for k, v in changes.items():
            print(f"    {k} = {v!r}")

        try:
            save_answer = input("\n  Save changes? [Y/n]: ").strip().lower()

        except EOFError:
            save_answer = "n"

        except KeyboardInterrupt:
            print("\n  Setup cancelled — changes discarded.")

            return {}

        if save_answer in ("n", "no"):
            print("  Discarded.")

            return {}

    else:
        print("\n  No changes made.")

    return changes
