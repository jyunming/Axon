"""REPL, display helpers, slash-command UI, and @file expansion for Axon."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from axon.main import AxonBrain

from axon.cli import _print_project_tree  # noqa: E402
from axon.collection_ops import clear_active_project  # noqa: E402
from axon.embeddings import OpenEmbedding  # noqa: E402
from axon.llm import OpenLLM, _copilot_device_flow, _fetch_copilot_models  # noqa: E402
from axon.rerank import OpenReranker  # noqa: E402
from axon.sessions import (  # noqa: E402
    _list_sessions,
    _load_session,
    _new_session,
    _print_sessions,
    _save_session,
)
from axon.vector_store import MultiVectorStore  # noqa: E402

_SLASH_COMMANDS = [
    "/clear",
    "/compact",
    "/context",
    "/discuss",
    "/embed ",
    "/exit",
    "/graph ",
    "/graph-viz",
    "/help",
    "/ingest ",
    "/keys",
    "/list",
    "/llm ",
    "/model ",
    "/project ",
    "/pull ",
    "/quit",
    "/rag ",
    "/refresh",
    "/resume ",
    "/retry",
    "/search",
    "/sessions",
    "/share ",
    "/stale",
    "/store ",
    "/vllm-url ",
]


def _save_env_key(env_name: str, key: str) -> None:
    """Persist *key* to ~/.axon/.env and os.environ."""
    _env_file = Path.home() / ".axon" / ".env"
    _env_file.parent.mkdir(parents=True, exist_ok=True)
    existing = _env_file.read_text(encoding="utf-8") if _env_file.exists() else ""
    lines = [ln for ln in existing.splitlines() if not ln.startswith(f"{env_name}=")]
    lines.append(f"{env_name}={key}")
    _env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ[env_name] = key


def _prompt_key_if_missing(provider: str, brain) -> bool:
    """If *provider* needs an API key/token and none is set, prompt the user.

    For github_copilot: runs the OAuth device flow (browser-based).
    For other providers: prompts for the API key via getpass.
    Saves to ~/.axon/.env and patches brain.config in-place.
    Returns True when a key is available (already set or just obtained).
    """
    _key_map = {
        "gemini": ("gemini_api_key", "GEMINI_API_KEY"),
        "ollama_cloud": ("ollama_cloud_key", "OLLAMA_CLOUD_KEY"),
        "openai": ("api_key", "OPENAI_API_KEY"),
        "github_copilot": ("copilot_pat", "GITHUB_COPILOT_PAT"),
    }
    if provider not in _key_map:
        return True
    attr, env_name = _key_map[provider]
    if getattr(brain.config, attr, ""):
        return True  # already configured

    if provider == "github_copilot":
        print("  ⚠️  No GitHub OAuth token set for the 'github_copilot' provider.")
        print("  Starting GitHub OAuth device flow…")
        try:
            key = _copilot_device_flow()
        except (EOFError, KeyboardInterrupt, RuntimeError) as e:
            print(f"\n  Cancelled: {e}")
            return False
        _save_env_key(env_name, key)
        brain.config.copilot_pat = key
        # Clear cached session so the new OAuth token is exchanged on next use
        brain.llm._openai_clients.pop("_copilot", None)
        brain.llm._openai_clients.pop("_copilot_session", None)
        brain.llm._openai_clients.pop("_copilot_token", None)
        print(f"  {env_name} saved and applied.")
        return True

    print(f"  ⚠️  No {env_name} set for the '{provider}' provider.")
    try:
        import getpass

        key = getpass.getpass(f"  Enter {env_name} (hidden, Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Skipped.")
        return False
    if not key:
        print("  Skipped — you can set it later with /keys set " + provider)
        return False
    _save_env_key(env_name, key)
    setattr(brain.config, attr, key)
    print(f"  {env_name} saved and applied.")
    return True


# ---------------------------------------------------------------------------
# GitHub Copilot OAuth helpers
# ---------------------------------------------------------------------------
# GitHub Copilot does NOT accept Personal Access Tokens.  The required flow is:
#   1. OAuth device flow  →  GitHub OAuth token  (stored in ~/.axon/.env)
#   2. OAuth token        →  Copilot session token  (in-memory, ~30 min TTL)
#   3. Session token used as Bearer on every API call.
# ---------------------------------------------------------------------------


def _make_completer(brain: AxonBrain):
    """Return a readline completer for slash commands, paths, and model names."""

    def completer(text: str, state: int):
        try:
            import readline

            full_line = readline.get_line_buffer()

            # Completing a slash command name
            if full_line.startswith("/") and " " not in full_line:
                matches = [c for c in _SLASH_COMMANDS if c.startswith(full_line)]
                return matches[state] if state < len(matches) else None

            # /ingest <path|glob> — complete filesystem paths
            if full_line.startswith("/ingest "):
                path_prefix = full_line[len("/ingest ") :]
                import glob as _glob

                matches = _glob.glob(path_prefix + "*")
                # Append / to directories
                matches = [m + "/" if os.path.isdir(m) else m for m in matches]
                return matches[state] if state < len(matches) else None

            # /model or /pull — complete model names
            if full_line.startswith("/model ") or full_line.startswith("/pull "):
                model_prefix = full_line.split(" ", 1)[1]
                # Explicit github_copilot/ prefix typed
                if model_prefix.startswith("github_copilot/") or model_prefix in (
                    "github_copilot",
                    "github_copilot/",
                ):
                    cp_prefix = model_prefix[len("github_copilot/") :]
                    cp_models = _fetch_copilot_models(brain.llm)
                    matches = [f"github_copilot/{m}" for m in cp_models if m.startswith(cp_prefix)]
                    return matches[state] if state < len(matches) else None
                # Active provider is github_copilot — complete bare model names
                if brain.config.llm_provider == "github_copilot" and "/" not in model_prefix:
                    cp_models = _fetch_copilot_models(brain.llm)
                    matches = [m for m in cp_models if m.startswith(model_prefix)]
                    return matches[state] if state < len(matches) else None
                try:
                    import ollama as _ollama

                    response = _ollama.list()
                    all_models = (
                        response.models
                        if hasattr(response, "models")
                        else response.get("models", [])
                    )
                    names = [
                        m.model if hasattr(m, "model") else m.get("name", "") for m in all_models
                    ]
                    matches = [n for n in names if n.startswith(model_prefix)]
                    return matches[state] if state < len(matches) else None
                except Exception:
                    return None

        except Exception:
            return None
        return None

    return completer


_MODEL_CTX: dict[str, int] = {
    "gemma": 8192,
    "gemma:2b": 8192,
    "gemma:7b": 8192,
    "llama3.1": 131072,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "mistral": 32768,
    "mistral:7b": 32768,
    "phi3": 131072,
    "phi3:mini": 131072,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
}


def _infer_provider(model: str) -> str:
    """Guess LLM provider from model name.

    Returns "gemini" for gemini-* models, "openai" for gpt-*/o1-*/o3-*/o4-*
    models (without a colon, since Ollama uses name:tag format), and "ollama"
    for everything else (local models, including gpt-oss:tag Ollama models).
    """
    m = model.lower()
    if m.startswith("gemini-"):
        return "gemini"
    # OpenAI model names never contain ':'; Ollama uses name:tag format.
    if ":" not in m and m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return "ollama"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return max(1, len(text) // 4)


def _token_bar(used: int, total: int, width: int = 20) -> str:
    """Return a visual fill bar: ████░░░░ 2,340 / 8,192 (28%)."""
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "[ok]" if pct < 0.6 else ("[!]" if pct < 0.85 else "[!!]")
    return f"{color} {bar}  {used:,} / {total:,} ({int(pct*100)}%)"


def _show_context(
    brain: AxonBrain,
    chat_history: list,
    last_sources: list,
    last_query: str,
) -> None:
    """Display a formatted context window panel with token usage, model info, and chat history.

    Shows:
    - Model info: LLM provider/model and context window size; embedding provider/model
    - Token usage: Rough estimates (4 chars/token) with visual bar and color indicator
    - RAG settings: top_k, similarity_threshold, hybrid_search, rerank, hyde, multi_query toggles
    - Chat history: Last 10 turns (user/assistant messages)
    - Last retrieved sources: Up to 8 chunks with similarity scores and source names
    - System prompt: Full text (word-wrapped)

    All content is wrapped in a box with section separators for readability.

    Args:
        brain: AxonBrain instance to extract model and config info.
        chat_history: List of message dicts {"role": "user"|"assistant", "content": str}.
        last_sources: List of document dicts from last retrieval (with "vector_score", "metadata", "text").
        last_query: The user query that was used for the last retrieval.
    """
    W = _box_width()  # match main header box width
    TOP = f"  ╭{'─' * W}╮"
    BOTTOM = f"  ╰{'─' * W}╯"
    SEP = f"  ├{'─' * W}┤"
    BLANK = f"  │{' ' * W}│"

    def _wlen(s: str) -> int:
        """Terminal display width: wide Unicode chars (emoji, CJK) count as 2 columns."""
        extra = sum(
            1
            for c in s
            if "\U00001100" <= c <= "\U00001fff"
            or "\U00002e80" <= c <= "\U00002eff"
            or "\U00002f00" <= c <= "\U00002fdf"
            or "\U00003000" <= c <= "\U00003fff"
            or "\U00004e00" <= c <= "\U00009fff"
            or "\U0000a000" <= c <= "\U0000abff"
            or "\U0000ac00" <= c <= "\U0000d7ff"
            or "\U0000f900" <= c <= "\U0000faff"
            or "\U0000fe10" <= c <= "\U0000fe1f"
            or "\U0000fe30" <= c <= "\U0000fe4f"
            or "\U0000ff00" <= c <= "\U0000ff60"
            or "\U0000ffe0" <= c <= "\U0000ffe6"
            or "\U0001f000" <= c <= "\U0001ffff"
            or "\U00020000" <= c <= "\U0002ffff"
        )
        return len(s) + extra

    def row(text: str = "", indent: int = 4) -> str:
        content = " " * indent + text
        display_w = _wlen(content)
        if display_w > W:
            content = content[: W - 1] + "…"
            display_w = _wlen(content)
        pad = W - display_w
        return f"  │{content}{' ' * pad}│"

    def section(title: str) -> str:
        content = f"  ▸  {title}"
        pad = W - _wlen(content)
        return f"  │{content}{' ' * pad}│"

    def wrap_row(text: str, indent: int = 4, max_lines: int = 0) -> list:
        """Word-wrap text into multiple box rows. 0 = no limit."""
        avail = W - indent - 2  # 2-char right margin so text never crowds the border
        words = text.split()
        lines_out, current = [], ""
        for w in words:
            if len(current) + len(w) + (1 if current else 0) <= avail:
                current = f"{current} {w}" if current else w
            else:
                if current:
                    lines_out.append(row(current, indent))
                current = w
        if current:
            lines_out.append(row(current, indent))
        return lines_out if not max_lines else lines_out[:max_lines]

    # ── Token estimates ────────────────────────────────────────────────────────
    system_text = brain._build_system_prompt(False)
    sys_tokens = _estimate_tokens(system_text)
    hist_tokens = sum(_estimate_tokens(m["content"]) for m in chat_history)
    src_tokens = sum(_estimate_tokens(s.get("text", "")) for s in last_sources)
    total_used = sys_tokens + hist_tokens + src_tokens

    model_key = brain.config.llm_model.split(":")[0].lower()
    ctx_size = _MODEL_CTX.get(brain.config.llm_model, _MODEL_CTX.get(model_key, 8192))
    remaining = max(0, ctx_size - total_used)
    pct = min(total_used / ctx_size, 1.0) if ctx_size > 0 else 0
    bar_w = 40
    filled = int(pct * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    indicator = "[ok]" if pct < 0.6 else ("[!]" if pct < 0.85 else "[!!]")

    lines = [TOP, BLANK]
    lines.append(row("Context Window", indent=4))
    lines.append(BLANK)

    # ── Model section ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Model"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(
            f"LLM    ·  {brain.config.llm_provider}/{brain.config.llm_model}"
            f"   ({ctx_size:,} token context window)"
        )
    )
    lines.append(row(f"Embed  ·  {brain.config.embedding_provider}/{brain.config.embedding_model}"))
    lines.append(BLANK)

    # ── Token usage ───────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Token Usage  (rough estimate — ~4 chars/token)"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(f"{indicator} {bar}  {total_used:,} / {ctx_size:,}  ({int(pct*100)}%)", indent=4)
    )
    lines.append(BLANK)
    lines.append(row(f"{'System prompt':<22}{sys_tokens:>7,} tokens"))
    lines.append(
        row(f"{'Chat history':<22}{hist_tokens:>7,} tokens    ({len(chat_history) // 2} turns)")
    )
    lines.append(
        row(f"{'Retrieved context':<22}{src_tokens:>7,} tokens    ({len(last_sources)} chunks)")
    )
    lines.append(row("─" * 40))
    lines.append(row(f"{'Total':<22}{total_used:>7,} tokens    ({remaining:,} remaining)"))
    lines.append(BLANK)

    # ── RAG settings ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("RAG Settings"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(
            f"top-k · {brain.config.top_k}    "
            f"threshold · {brain.config.similarity_threshold}    "
            f"hybrid · {'ON' if brain.config.hybrid_search else 'OFF'}    "
            f"rerank · {'ON' if brain.config.rerank else 'OFF'}    "
            f"hyde · {'ON' if brain.config.hyde else 'OFF'}    "
            f"multi-query · {'ON' if brain.config.multi_query else 'OFF'}"
        )
    )
    lines.append(BLANK)

    # ── Chat history ───────────────────────────────────────────────────────────
    lines.append(SEP)
    turns = len(chat_history) // 2
    lines.append(section(f"Chat History  ({turns} turns)"))
    lines.append(SEP)
    lines.append(BLANK)
    if not chat_history:
        lines.append(row("(empty)"))
    else:
        shown = chat_history[-10:]
        for msg in shown:
            tag = "You   " if msg["role"] == "user" else "Brain "
            snip = msg["content"].replace("\n", " ")
            avail = W - 14
            if len(snip) > avail:
                snip = snip[:avail] + "…"
            lines.append(row(f"{tag}  {snip}"))
        if len(chat_history) > 10:
            lines.append(row(f"… {len(chat_history) - 10} earlier messages not shown"))
    lines.append(BLANK)

    # ── Last retrieved sources ─────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section(f"Last Retrieved Sources  ({len(last_sources)} chunks)"))
    lines.append(SEP)
    lines.append(BLANK)
    if last_query:
        lines.append(row(f'query · "{last_query[:W - 14]}"'))
        lines.append(BLANK)
    if not last_sources:
        lines.append(row("(no retrieval yet)"))
    else:
        for i, src in enumerate(last_sources[:8], 1):
            meta = src.get("metadata", {})
            name = os.path.basename(meta.get("source", src.get("id", "?")))
            score = src.get("vector_score", src.get("score", 0))
            kind = "web" if src.get("is_web") else "doc"
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(row(f"{i:>2}. {kind} {score_bar} {score:.3f}   {name}"))
    lines.append(BLANK)

    # ── System prompt ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("System Prompt"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.extend(wrap_row(system_text.replace("\n", " "), indent=4))
    lines.append(BLANK)
    lines.append(BOTTOM)

    for line in lines:
        print(line)
    print()


def _do_compact(brain: AxonBrain, chat_history: list) -> None:
    """Summarize chat history via LLM and replace it with a single summary turn.

    Condenses all messages in chat_history into a 4-6 sentence summary using the configured LLM.
    The original conversation is replaced with a single message prefixed with
    "[Conversation summary]: " to preserve context while freeing up token space.

    If chat_history is empty, prints a message and returns without action.

    Args:
        brain: AxonBrain instance used to call the LLM for summarization.
        chat_history: List of message dicts to summarize (modified in-place; emptied and refilled with summary).
    """
    if not chat_history:
        print("  Nothing to compact — chat history is empty.")
        return

    turns_before = len(chat_history)
    print(f"  ⠿ Compacting {turns_before} turns…", end="", flush=True)

    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history
    )
    summary_prompt = (
        "Summarize the following conversation in 4-6 concise sentences. "
        "Preserve all key facts, decisions, and topics discussed. "
        "Write in third person ('The user asked…'). "
        "Output only the summary, no preamble.\n\n"
        f"{conversation}"
    )
    try:
        summary = brain.llm.complete(summary_prompt, system_prompt=None, chat_history=[])
        chat_history.clear()
        chat_history.append({"role": "assistant", "content": f"[Conversation summary]: {summary}"})
        tokens_saved = _estimate_tokens(conversation) - _estimate_tokens(summary)
        print(f"\r  Compacted {turns_before} turns -> 1 summary  (~{tokens_saved:,} tokens freed)")
    except Exception as e:
        print(f"\r  Compact failed: {e}")


# ── Banner constants ───────────────────────────────────────────────────────────
_HINT = "  Type your question  ·  /help for commands  ·  Tab to autocomplete  ·  @file or @folder/ to attach context"


def _box_width() -> int:
    """Return inner box width: terminal columns minus 4, minimum 43."""
    return max(43, shutil.get_terminal_size((120, 24)).columns - 4)


# FIGlet "Big" ASCII art for AXON — all chars are 1-col wide, each line is 35 cols
_AXON_ART = [
    " █████╗ ██╗  ██╗ ██████╗ ███╗   ██╗",
    "██╔══██╗╚██╗██╔╝██╔═══██╗████╗  ██║",
    "███████║ ╚███╔╝ ██║   ██║██╔██╗ ██║",
    "██╔══██║ ██╔██╗ ██║   ██║██║╚██╗██║",
    "██║  ██║██╔╝ ██╗╚██████╔╝██║ ╚████║",
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
]
# 24-bit blue gradient: light sky → cornflower → dodger → royal → medium → cobalt
_AXON_BLUE = [
    "\x1b[38;2;173;216;230m",  # light blue
    "\x1b[38;2;135;206;250m",  # light sky blue
    "\x1b[38;2;100;149;237m",  # cornflower blue
    "\x1b[38;2;30;144;255m",  # dodger blue
    "\x1b[38;2;65;105;225m",  # royal blue
    "\x1b[38;2;0;71;171m",  # cobalt blue
]
_AXON_RST = "\x1b[0m"


# Symmetrical brain/axon hub design (25 columns wide)
_BRAIN_ART = [
    "(O)~~.             .~~(O)",
    "  \\   *~._     _.~*   /  ",
    "[#]--.    ( O )    .--[#]",
    "[#]--'    ( O )    '--[#]",
    "  /   *~.'     '.~*   \\  ",
    "(O)~~'             '~~(O)",
]


def _get_brain_anim_row(row_idx: int, frame: int, width: int) -> str:
    """Return one row of the animated brain design, centered in `width`."""
    if width < 25:
        return " " * width

    pad = (width - 25) // 2
    l_pad = " " * pad
    r_pad = " " * (width - 25 - pad)

    line = _BRAIN_ART[row_idx]
    RST = "\x1b[0m"
    DIM = "\x1b[38;2;100;110;130m"  # Muted steel-blue (base connection)
    PULSE = "\x1b[38;2;180;210;255m"  # Pastel Blue (moving signal)
    GLOW = "\x1b[38;2;255;255;255m"  # Soft White (peak flash)

    # Staggered animation: 6 paths pulsing at different offsets in a 24-frame cycle
    total_cycle = 24
    # Paths: (phase, row, start, end)
    # Phase 0: Center, Phase 1: Inner path, Phase 2: Outer path, Phase 3: Cell/Dot
    tl_path = [(1, 1, 6, 10), (2, 0, 3, 6), (3, 0, 0, 3), (3, 1, 2, 3)]
    tr_path = [(1, 1, 15, 19), (2, 0, 19, 22), (3, 0, 22, 25), (3, 1, 22, 23)]
    ml_path = [(1, 2, 3, 6), (1, 3, 3, 6), (3, 2, 0, 3), (3, 3, 0, 3)]  # Mid-left
    mr_path = [(1, 2, 19, 22), (1, 3, 19, 22), (3, 2, 22, 25), (3, 3, 22, 25)]  # Mid-right
    bl_path = [(1, 4, 6, 10), (2, 5, 3, 6), (3, 5, 0, 3), (3, 4, 2, 3)]
    br_path = [(1, 4, 15, 19), (2, 5, 19, 22), (3, 5, 22, 25), (3, 4, 22, 23)]

    # (offset, path_segments)
    groups = [
        (0, tl_path),
        (4, br_path),
        (8, tr_path),
        (12, bl_path),
        (16, ml_path),
        (20, mr_path),
    ]

    char_levels = [0] * 25  # 0: DIM, 1: PULSE, 2: GLOW
    if frame == -1:
        # Fully connected state: all paths lit, center and cells glowing
        for _, path in groups:
            for phase, r, s, e in path + [(0, 2, 10, 15), (0, 3, 10, 15)]:
                if r == row_idx:
                    level = 2 if phase in (0, 3) else 1
                    for i in range(s, e):
                        char_levels[i] = max(char_levels[i], level)
    else:
        for offset, path in groups:
            rel = (frame - offset) % total_cycle
            # Add center to every path's start (phase 0)
            for phase, r, s, e in path + [(0, 2, 10, 15), (0, 3, 10, 15)]:
                if r == row_idx:
                    level = 0
                    if rel == phase:
                        level = 2
                    elif rel == (phase + 1) % total_cycle:
                        level = 1
                    for i in range(s, e):
                        char_levels[i] = max(char_levels[i], level)

    char_colors = [DIM] * 25
    for i, level in enumerate(char_levels):
        if level == 2:
            char_colors[i] = GLOW
        elif level == 1:
            char_colors[i] = PULSE

    res = ""
    curr_col = ""
    for i, char in enumerate(line):
        if char == " ":
            res += char
            continue
        if char_colors[i] != curr_col:
            res += char_colors[i]
            curr_col = char_colors[i]
        res += char
    return l_pad + res + RST + r_pad


def _brow(content: str, emoji_extra: int = 0) -> str:
    """One box row: pads/truncates content to exactly _box_width() terminal columns."""
    bw = _box_width()
    vis = len(content) + emoji_extra
    if vis > bw:
        content = content[: bw - emoji_extra - 1] + "…"
        vis = bw
    pad = bw - vis
    return f"  ┃{content}{' ' * pad}┃"


def _anim_pad(row_idx: int, frame: int, width: int) -> str:
    """Return an animated brain design centered in `width`."""
    return _get_brain_anim_row(row_idx, frame, width)


def _build_header(brain: AxonBrain, tick_lines: list | None = None) -> list:
    """Return lines of the pinned header box (airy layout)."""
    model_s = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    embed_s = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
    search_s = "ON  (Brave Search)" if brain.config.truth_grounding else "OFF"
    discuss_s = "ON" if brain.config.discussion_fallback else "OFF"
    hybrid_s = "ON" if brain.config.hybrid_search else "OFF"
    topk_s = str(brain.config.top_k)
    thr_s = str(brain.config.similarity_threshold)
    try:
        docs = brain.list_documents()
        doc_s = f"{sum(d['chunks'] for d in docs)} chunks  ({len(docs)} files)"
    except Exception:
        doc_s = "unknown"

    bw = _box_width()
    apad_w = max(0, bw - 39)  # 4 indent + 35 art cols = 39 vis cols

    # Build tick status — wrap onto a second row if too wide
    tick_items = [f"✓ {t}" for t in tick_lines] if tick_lines else ["✓ Ready"]
    ticks_s = "   ".join(tick_items)
    inner_w = bw - 4  # 4-char left indent "    "
    if len(ticks_s) > inner_w:
        # Split into two roughly equal halves at a separator boundary
        mid = len(tick_items) // 2
        ticks_s = "   ".join(tick_items[:mid])
        ticks_s2 = "   ".join(tick_items[mid:])
    else:
        ticks_s2 = None

    blank = f"  ┃{' ' * bw}┃"
    rows = [
        f"  \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m",  # 1
        blank,  # 2
        *[  # 3-8  blue-shaded art lines + brain design
            f"  ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{_get_brain_anim_row(i, -1, apad_w)}┃"
            for i, line in enumerate(_AXON_ART)
        ],
        blank,  # 9
        _brow(f"    LLM    ·  {model_s}"),  # 6
        _brow(f"    Embed  ·  {embed_s}"),  # 7
        blank,  # 8
        _brow(f"    Search ·  {search_s:<26}  Discuss  ·  {discuss_s}"),  # 9
        _brow(
            f"    Docs   ·  {doc_s:<26}  Hybrid   ·  {hybrid_s}   top-k · {topk_s}   threshold · {thr_s}"
        ),  # 10
        blank,  # 11
        _brow(f"    {ticks_s}"),  # 12
    ]
    if ticks_s2:
        rows.append(_brow(f"    {ticks_s2}"))  # 12b (overflow)
    rows.append(f"  \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m")  # 13
    return rows


def _draw_header(brain: AxonBrain, tick_lines: list | None = None) -> None:
    """Clear screen and draw the welcome header box with LLM and embedding model info.

    Displays initialization status lines (e.g., "✓ Embedding ready [CPU]", "✓ BM25 · 42 docs").
    Clears the entire screen and redraws the header with hints for available REPL commands.
    Uses ANSI codes to clear and position the cursor — no scroll region (natural terminal scrollback).

    Args:
        brain: AxonBrain instance to extract model and provider information.
        tick_lines: Optional list of status messages (e.g., ["Starting", "Embedding ready [CPU]"])
                   to display in the header box.
    """
    bw = _box_width()
    sep = "  " + "─" * (bw + 2)
    lines = _build_header(brain, tick_lines)
    sys.stdout.write("\033[2J\033[H")  # clear screen, cursor to top-left
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.write("\n" + _HINT + "\n" + sep + "\n\n")
    sys.stdout.flush()


def _print_recent_turns(history: list, n_turns: int = 2) -> None:
    """Print the last n_turns of Q&A below the header so context is visible.

    Args:
        history: chat_history list of {"role": ..., "content": ...} dicts.
        n_turns: Number of complete Q&A turns to show (each turn = 1 user + 1 assistant message).
    """
    if not history:
        return
    recent = history[-(n_turns * 2) :]
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            sys.stdout.write(f"  \033[1;32mYou\033[0m: {content}\n")
        elif role == "assistant":
            # Cap very long responses so they don't flood the screen
            if len(content) > 600:
                content = content[:600] + "…"
            sys.stdout.write(f"\n  \033[1;33mAxon\033[0m:\n  {content}\n")
        sys.stdout.write("\n")
    sys.stdout.flush()


class _InitDisplay(logging.Handler):
    """Intercepts initialization log messages and renders animated status in a box.

    Displays a 7-line box with title and status line updated in-place using ANSI cursor positioning.
    Uses a braille spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) that rotates every 0.08 seconds.
    Collects completed steps as checkmarks (✓) for the final banner display.

    The box is printed once at initialization, then the step line (line 5) is updated in-place
    as different initialization phases complete (Starting, Loading models, Vector store ready, etc.).
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        super().__init__()
        self._step: str = ""
        self._idx: int = 0
        self._anim_frame: int = 0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.tick_lines: list = []  # collected for the final banner
        # Print CLOSED 7-line box immediately — step line updated in-place
        bw = _box_width()
        art_pad = " " * max(0, bw - 39)  # 4 indent + 35 art cols = 39 vis cols
        _art_rows = "".join(
            f"  ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{art_pad}┃\n"
            for i, line in enumerate(_AXON_ART)
        )
        sys.stdout.write(
            f"\n  \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m\n"
            f"  ┃{' ' * bw}┃\n" + _art_rows + f"  ┃{' ' * bw}┃\n"
            f"  ┃{'    ⠿  Initializing…'.ljust(bw)}┃\n"  # step line (3rd from bottom)
            f"  ┃{' ' * bw}┃\n"
            f"  \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m\n"
        )
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    def _spin_loop(self) -> None:
        while not self._done.wait(0.08):
            with self._lock:
                self._anim_frame += 1
                bw = _box_width()
                apad_w = max(0, bw - 39)

                # Rebuild the 6 animated art rows (AXON text + signal animation)
                art_rows = [
                    f"  ┃    {_AXON_BLUE[i]}{_AXON_ART[i]}{_AXON_RST}"
                    f"{_anim_pad(i, self._anim_frame, apad_w)}┃"
                    for i in range(6)
                ]

                # Box line layout (1-indexed, cursor rests at line 14 after init print):
                #  1=blank  2=╭╮  3=┃blank┃  4-9=art  10=┃blank┃  11=┃step┃  12=┃blank┃  13=╰╯
                # Go up 10 from line 14 → line 4 (first art row), rewrite all 6.
                sys.stdout.write("\033[10A")
                for arow in art_rows:
                    sys.stdout.write(f"\r{arow}\n")
                # Cursor now at line 10 (blank row after art).

                if self._step:
                    spinner = self._FRAMES[self._idx % len(self._FRAMES)]
                    line = _brow(f"    {spinner}  {self._step}")
                    # Down 1 → line 11 (step), write, newline → 12, down 2 → 14.
                    sys.stdout.write(f"\033[1B\r{line}\n\033[2B")
                    self._idx += 1
                else:
                    # Skip blank(10) step(11) blank(12) bottom(13) → back to line 14.
                    sys.stdout.write("\033[4B")

                sys.stdout.flush()

    def _tick(self, label: str) -> None:
        with self._lock:
            self._step = ""
            self.tick_lines.append(label)
            line = _brow(f"    ✓  {label}")
            sys.stdout.write(f"\033[3A\r{line}\n\033[2B")
            sys.stdout.flush()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Initializing Axon" in msg:
            with self._lock:
                self._step = "Starting…"
        elif "Loading Sentence Transformers" in msg:
            m = re.search(r":\s*(.+)$", msg)
            with self._lock:
                self._step = f"Loading {m.group(1).strip() if m else 'model'}…"
        elif "Use pytorch device_name" in msg:
            m = re.search(r":\s*(.+)$", msg)
            self._tick(f"Embedding ready  [{m.group(1).strip() if m else 'cpu'}]")
        elif "Initializing ChromaDB" in msg:
            with self._lock:
                self._step = "Vector store…"
        elif "Loaded BM25 corpus" in msg:
            m = re.search(r"(\d+) documents", msg)
            self._tick("Vector store ready")
            self._tick(f"BM25  ·  {m.group(1) if m else '?'} docs")
        elif "Axon ready" in msg:
            self._done.set()

    def stop(self) -> None:
        self._done.set()
        with self._lock:
            self._step = ""
        self._thread.join(timeout=0.5)


_AT_TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".htm",
    ".css",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".xml",
    ".ini",
    ".cfg",
    ".env",
    ".tf",
    ".proto",
    ".graphql",
    ".jsonl",
    ".ndjson",
    ".ipynb",
}
# Extensions handled by dedicated loaders (extract clean text from binary formats)
_AT_LOADER_EXTS = {
    ".docx",
    ".pptx",
    ".pdf",
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
    ".xlsx",
    ".xls",
    ".parquet",
    ".epub",
    ".rtf",
    ".eml",
    ".msg",
    ".tex",
}
_AT_DIR_MAX_BYTES = 120_000  # ~120 KB total across all files in a folder
_AT_FILE_MAX_BYTES = 40_000  # ~40 KB per single file


def _expand_at_files(text: str) -> str:
    """Expand @path references in user input with file/folder contents (read-only).

    - @file.txt / @file.docx / @file.pdf → inlines extracted text
    - @folder/   → recursively reads all supported files in the folder (capped at
                   _AT_DIR_MAX_BYTES total; unsupported / oversized files are skipped)
    """
    # Imported at call time so tests can patch axon.loaders.DOCXLoader / PDFLoader.
    from axon.loaders import DOCXLoader, PDFLoader, PPTXLoader  # noqa: F401 (used in _loader_map)

    def _read_text_file(path: str, max_bytes: int = _AT_FILE_MAX_BYTES) -> str:
        try:
            with open(path, "rb") as f:
                raw = f.read(max_bytes)
            text = raw.decode("utf-8", errors="ignore")
            truncated = os.path.getsize(path) > max_bytes
            return text + ("\n… (truncated)" if truncated else "")
        except OSError:
            return ""

    def _read_via_loader(path: str) -> str:
        from axon.loaders import (
            EMLLoader,
            EPUBLoader,
            ExcelLoader,
            LaTeXLoader,
            MSGLoader,
            ParquetLoader,
            RTFLoader,
        )

        _loader_map = {
            ".docx": DOCXLoader,
            ".pptx": PPTXLoader,
            ".xlsx": ExcelLoader,
            ".xls": ExcelLoader,
            ".parquet": ParquetLoader,
            ".epub": EPUBLoader,
            ".rtf": RTFLoader,
            ".eml": EMLLoader,
            ".msg": MSGLoader,
            ".tex": LaTeXLoader,
        }
        try:
            ext = os.path.splitext(path)[1].lower()
            loader_cls = _loader_map.get(ext)
            loader = loader_cls() if loader_cls else PDFLoader()
            docs = loader.load(path)
            chunks = [d.get("text", "") for d in docs if d.get("text")]
            joined = "\n\n".join(chunks)
            if len(joined) > _AT_FILE_MAX_BYTES:
                joined = joined[:_AT_FILE_MAX_BYTES] + "\n… (truncated)"
            return joined
        except Exception as e:
            return f"(could not extract text: {e})"

    def _read_file(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in _AT_LOADER_EXTS:
            return _read_via_loader(path)
        return _read_text_file(path)

    def _expand_dir(dirpath: str) -> str:
        supported = _AT_TEXT_EXTS | _AT_LOADER_EXTS
        parts: list[str] = []
        total = 0
        for root, dirs, files in os.walk(dirpath):
            dirs.sort()
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in supported:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, dirpath)
                if total >= _AT_DIR_MAX_BYTES:
                    # Budget exhausted — skip remaining files without reading them
                    parts.append(f"\n--- @{rel} (skipped: context limit reached) ---")
                    continue
                content = _read_file(fpath)
                if not content:
                    continue
                encoded = content.encode("utf-8", errors="ignore")
                next_size = len(encoded)
                if total + next_size > _AT_DIR_MAX_BYTES:
                    remaining = _AT_DIR_MAX_BYTES - total
                    truncated_text = encoded[:remaining].decode("utf-8", errors="ignore")
                    parts.append(f"\n--- @{rel} ---\n{truncated_text}\n… (truncated)\n--- end ---")
                    total = _AT_DIR_MAX_BYTES
                else:
                    parts.append(f"\n--- @{rel} ---\n{content}\n--- end ---")
                    total += next_size
        return "\n".join(parts) if parts else f"\n(no readable files found in {dirpath})"

    def _replace(m: re.Match) -> str:
        path = m.group(1).rstrip("/\\")
        if os.path.isdir(path):
            return f"\n\n=== folder: {path} ===\n{_expand_dir(path)}\n=== end folder ===\n"
        if os.path.isfile(path):
            content = _read_file(path)
            if content:
                return f"\n\n--- @{path} ---\n{content}\n--- end ---\n"
        return m.group(0)

    return re.sub(r"@(\S+)", _replace, text)


def _interactive_repl(
    brain: AxonBrain,
    stream: bool = True,
    init_display: _InitDisplay | None = None,
    quiet: bool = False,
) -> None:
    """Interactive REPL chat session with session persistence and live tab completion.

    Features:
    - Session persistence: auto-saves to ~/.axon/sessions/session_<timestamp>.json
    - Live tab completion: slash commands, filesystem paths, Ollama model names via prompt_toolkit
    - Animated spinners: braille spinner during init and LLM generation (disabled in quiet mode)
    - Slash commands: /help, /list, /ingest, /model, /embed, /pull, /search, /discuss, /rag,
      /compact, /context, /sessions, /resume, /retry, /clear, /project, /keys, /vllm-url, /quit, /exit
    - @file/folder context: type @path/file.txt or @path/folder/ to inline contents into your query (read-only)
    - Shell passthrough: !command runs a shell command (local-only by default)
    - Pinned status info: token usage, model info, RAG settings visible at terminal bottom

    Args:
        brain: AxonBrain instance to use for queries.
        stream: If True, streams LLM response token-by-token; if False, waits for full response.
        init_display: Optional _InitDisplay handler to stop after initialization.
        quiet: Suppress spinners and progress bars (auto-enabled for non-TTY stdin).
    """
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="google")

    # Silence INFO logs — they clutter the interactive UI
    import logging as _logging

    for _log in (
        "Axon",
        "Axon.Retrievers",
        "httpx",
        "sentence_transformers",
        "chromadb",
        "httpcore",
    ):
        _lg = _logging.getLogger(_log)
        _lg.setLevel(_logging.WARNING)
        _lg.propagate = False  # prevent bubbling to root logger

    # ── Input: prefer prompt_toolkit (live completions), fall back to readline ──
    _pt_session = None
    try:
        import glob as _pglob

        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.formatted_text import ANSI as _PTANSI
        from prompt_toolkit.formatted_text import HTML as _PThtml
        from prompt_toolkit.formatted_text import FormattedText as _PTFT  # noqa: F401
        from prompt_toolkit.history import FileHistory as _FileHistory
        from prompt_toolkit.styles import Style

        _HIST_DIR = os.path.expanduser("~/.axon")
        os.makedirs(_HIST_DIR, exist_ok=True)
        _HIST_FILE = os.path.join(_HIST_DIR, "repl_history")

        _PT_STYLE = Style.from_dict(
            {
                "": "",
                "completion-menu.completion.current": "bg:#444466 #ffffff",
                "bottom-toolbar": "bg:#0a2a5e #c8d8f0",
            }
        )

        class _PTCompleter(Completer):
            def __init__(self, brain_ref):
                self._brain = brain_ref

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                # ── slash command name ─────────────────────────────────────
                if text.startswith("/") and " " not in text:
                    for cmd in _SLASH_COMMANDS:
                        c = cmd.rstrip()
                        if c.startswith(text):
                            yield Completion(c[len(text) :], display=c, display_meta="command")
                # ── /ingest path / glob ───────────────────────────────────
                elif text.startswith("/ingest "):
                    prefix = text[len("/ingest ") :]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(p[len(prefix) :], display=disp)
                # ── /model <provider/model> ───────────────────────────────
                elif text.startswith("/model ") or text.startswith("/embed "):
                    cmd_len = len("/model ") if text.startswith("/model ") else len("/embed ")
                    prefix = text[cmd_len:]
                    if prefix.startswith("github_copilot/") or prefix in (
                        "github_copilot",
                        "github_copilot/",
                    ):
                        cp_prefix = prefix[len("github_copilot/") :]
                        for mid in _fetch_copilot_models(self._brain.llm):
                            if mid.startswith(cp_prefix):
                                full = f"github_copilot/{mid}"
                                yield Completion(
                                    full[len(prefix) :], display=full, display_meta="copilot"
                                )
                    elif self._brain.config.llm_provider == "github_copilot" and "/" not in prefix:
                        # Active provider is github_copilot — complete bare model names
                        for mid in _fetch_copilot_models(self._brain.llm):
                            if mid.startswith(prefix):
                                yield Completion(
                                    mid[len(prefix) :], display=mid, display_meta="copilot"
                                )
                    else:
                        try:
                            import ollama as _ol

                            resp = _ol.list()
                            mods = (
                                resp.models if hasattr(resp, "models") else resp.get("models", [])
                            )
                            for m in mods:
                                name = m.model if hasattr(m, "model") else m.get("name", "")
                                if name.startswith(prefix):
                                    yield Completion(name[len(prefix) :], display=name)
                        except Exception:
                            pass
                # ── /resume <session-id> ──────────────────────────────────
                elif text.startswith("/resume "):
                    prefix = text[len("/resume ") :]
                    for s in _list_sessions(project=brain._active_project):
                        sid = s["id"]
                        if sid.startswith(prefix):
                            turns = len(s.get("history", [])) // 2
                            yield Completion(
                                sid[len(prefix) :], display=sid, display_meta=f"{turns} turns"
                            )
                # ── /llm <option> ─────────────────────────────────────────
                elif text.startswith("/llm "):
                    opts = ["temperature "]
                    prefix = text[len("/llm ") :]
                    for o in opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                # ── /rag <option> ─────────────────────────────────────────
                elif text.startswith("/rag "):
                    opts = [
                        "topk ",
                        "threshold ",
                        "hybrid",
                        "rerank",
                        "rerank-model ",
                        "hyde",
                        "multi",
                        "step-back",
                        "decompose",
                        "compress",
                        "cite",
                        "raptor",
                        "graph-rag",
                    ]
                    prefix = text[len("/rag ") :]
                    for o in opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                # ── @file context attachment ──────────────────────────────
                elif "@" in text:
                    at_pos = text.rfind("@")
                    prefix = text[at_pos + 1 :]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(
                            p[len(prefix) :], display=disp, display_meta="file context"
                        )
                # ── /project <subcommand> ──────────────────────────────────
                elif text.startswith("/project "):
                    sub_opts = ["list", "new ", "switch ", "delete ", "folder"]
                    prefix = text[len("/project ") :]
                    for o in sub_opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                    # also complete project names for switch/delete
                    if text.startswith(("/project switch ", "/project delete ")):
                        cmd_len = (
                            len("/project switch ")
                            if text.startswith("/project switch ")
                            else len("/project delete ")
                        )
                        pfx = text[cmd_len:]
                        try:
                            from axon.projects import list_projects

                            for p in list_projects():
                                n = p["name"]
                                if n.startswith(pfx):
                                    yield Completion(n[len(pfx) :], display=n)
                        except Exception:
                            pass

        def _toolbar():
            def _t(s: str, w: int) -> str:
                return s if len(s) <= w else s[: w - 1] + "…"

            # No explicit background codes — the bottom-toolbar class paints
            # bg:#0a2a5e for every cell uniformly.  Bold labels only change
            # font weight; the class background is never overridden.
            _BON = "\x1b[1m"  # bold on
            _BOF = "\x1b[22m"  # bold off
            _RST = "\x1b[0m"

            def _lbl(text: str) -> str:
                return f"{_BON}{text}{_BOF}"

            def _pad(label: str, val: str, width: int) -> str:
                return " " * max(0, width - len(label) - 1 - len(str(val)))

            m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
            emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
            try:
                docs = brain.vector_store.collection.count()
                doc_s = f"{docs} chunks"
            except Exception:
                doc_s = "?"
            proj = getattr(brain, "_active_project", "default")
            _proj_display = proj.replace("/", " > ") if proj != "default" else ""
            _merged = isinstance(brain.vector_store, MultiVectorStore)
            _merged_tag = " [merged]" if _merged else ""
            proj_s = f"  │  {_proj_display}{_merged_tag}" if proj != "default" else ""
            sep = "  │  "
            W1, W2 = 28, 30
            C1 = len("LLM  ") + W1  # 33
            C2 = len("Embed  ") + W2  # 37
            s_state = "ON" if brain.config.truth_grounding else "off"
            d_state = "ON" if brain.config.discussion_fallback else "off"
            h_state = "ON" if brain.config.hybrid_search else "off"

            row1 = (
                f"  {_lbl('LLM')}  {_t(m, W1):{W1}}{sep}"
                f"{_lbl('Embed')}  {_t(emb, W2):{W2}}{sep}"
                f"{_lbl('Docs')}  {doc_s}"
            )
            row2 = (
                f"  {_lbl('search')}:{s_state}{_pad('search', s_state, C1)}{sep}"
                f"{_lbl('discuss')}:{d_state}{_pad('discuss', d_state, C2)}{sep}"
                f"{_lbl('hybrid')}:{h_state}  "
                f"{_lbl('top-k')}:{brain.config.top_k}  "
                f"{_lbl('thr')}:{brain.config.similarity_threshold}"
                f"{proj_s}"
            )
            return _PTANSI(f"{row1}\n{row2}{_RST}")

        _pt_session = PromptSession(
            completer=_PTCompleter(brain),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PT_STYLE,
            complete_while_typing=True,
            bottom_toolbar=_toolbar,
            history=_FileHistory(_HIST_FILE),
        )
    except ImportError:
        # Fall back to readline with history persistence
        try:
            import atexit
            import readline

            _hist_file = os.path.expanduser("~/.axon/repl_history")
            os.makedirs(os.path.dirname(_hist_file), exist_ok=True)
            try:
                readline.read_history_file(_hist_file)
            except FileNotFoundError:
                pass
            readline.set_history_length(2000)
            atexit.register(readline.write_history_file, _hist_file)
            readline.set_completer(_make_completer(brain))
            readline.set_completer_delims("")
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\C-l": clear-screen')
        except ImportError:
            pass

    def _print_status_bar() -> None:
        """Reprint the 2-row status bar to stdout after each response."""

        def _t(s: str, w: int) -> str:
            return s if len(s) <= w else s[: w - 1] + "…"

        m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
        emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
        try:
            docs = brain.vector_store.collection.count()
            doc_s = f"{docs} chunks"
        except Exception:
            doc_s = "?"
        s_val = "search:ON" if brain.config.truth_grounding else "search:off"
        d_val = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
        h_val = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
        tk = f"top-k:{brain.config.top_k}  thr:{brain.config.similarity_threshold}"
        proj = getattr(brain, "_active_project", "default")
        _proj_display = proj.replace("/", " > ") if proj != "default" else ""
        _merged = isinstance(brain.vector_store, MultiVectorStore)
        _merged_tag = " [merged]" if _merged else ""
        proj_s = f"  │  {_proj_display}{_merged_tag}" if proj != "default" else ""
        sep = "  │  "
        W1, W2 = 28, 30
        C1 = len("LLM  ") + W1  # 33
        C2 = len("Embed  ") + W2  # 37
        row1 = (
            f"  \033[1mLLM\033[0m\033[2m  {_t(m, W1):{W1}}"
            f"{sep}\033[0m\033[1mEmbed\033[0m\033[2m  {_t(emb, W2):{W2}}"
            f"{sep}\033[0m\033[1mDocs\033[0m\033[2m  {doc_s}\033[0m"
        )
        row2 = f"\033[2m  {s_val:<{C1}}" f"{sep}{d_val:<{C2}}" f"{sep}{h_val}  {tk}{proj_s}\033[0m"
        print(row1)
        print(row2)

    def _read_input(prompt: str = "") -> str:
        if _pt_session:
            _p = _PThtml("<ansigreen><b>You</b></ansigreen>: ") if not prompt else prompt
            return _pt_session.prompt(_p)
        return input(prompt if prompt else "\033[1;32mYou\033[0m: ")

    def _shell_passthrough_allowed() -> tuple[bool, str]:
        policy = str(getattr(brain.config, "repl_shell_passthrough", "local_only")).lower().strip()
        if policy == "always":
            return True, policy
        if policy == "off":
            return False, policy
        kind = getattr(brain, "_active_project_kind", "default")
        if not isinstance(kind, str):
            kind = "default"
        read_only_scope = getattr(brain, "_read_only_scope", False)
        if not isinstance(read_only_scope, bool):
            read_only_scope = False
        is_local_mode = kind in {"default", "local"} and not read_only_scope
        return is_local_mode, policy

    # REPL is conversational — always let the LLM answer even with no RAG hits
    brain.config.discussion_fallback = True

    _tick_lines = init_display.tick_lines if init_display else []
    _draw_header(brain, _tick_lines)

    # ── Session init ───────────────────────────────────────────────────────────
    session: dict = _new_session(brain)
    chat_history: list = session["history"]

    _last_sources: list = []
    _last_query: str = ""

    # Initial snapshot to avoid printing status on the very first query
    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
    tk = f"top-k:{brain.config.top_k}"
    thr = f"thr:{brain.config.similarity_threshold}"
    _last_config_snapshot: tuple = (m, s_v, d_v, h_v, tk, thr)

    while True:
        try:
            user_input = _read_input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # --- Shell passthrough: !command ---
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                allowed, policy = _shell_passthrough_allowed()
                if not allowed:
                    print(
                        "  Shell passthrough blocked by policy "
                        f"(repl.shell_passthrough={policy})."
                    )
                    continue
                import subprocess

                subprocess.run(shell_cmd, shell=True)
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Allow "/<cmd> help" as alias for "/help <cmd>"
            if arg.strip().lower() == "help" and cmd not in ("/help",):
                arg = cmd.lstrip("/")
                cmd = "/help"
            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/help":
                if arg:
                    # Per-command detail
                    _detail = {
                        "model": "  /model <model>              keep current provider\n"
                        "  /model <provider>/<model>   switch provider + model\n"
                        "  providers: ollama, gemini, openai, ollama_cloud, vllm, github_copilot\n"
                        "  e.g.  /model gemini/gemini-2.0-flash\n"
                        "        /model ollama/gemma:2b\n"
                        "        /model openai/gpt-4o\n"
                        "        /model vllm/meta-llama/Llama-3.1-8B-Instruct",
                        "embed": "  /embed <model>              keep current provider\n"
                        "  /embed <provider>/<model>   switch provider + model\n"
                        "  /embed /path/to/local        local HuggingFace folder\n"
                        "  providers: sentence_transformers, ollama, fastembed, openai\n"
                        "  !  Re-ingest after changing embedding model.",
                        "ingest": "  /ingest <path>              ingest a directory\n"
                        "  /ingest ./src/*.py           glob pattern\n"
                        "  /ingest ./notes/**/*.md      recursive glob",
                        "llm": "  /llm                         show LLM settings (provider, model, temperature)\n"
                        "  /llm temperature <0.0–2.0>   set generation temperature\n"
                        "  Lower temperature = more deterministic; higher = more creative.",
                        "rag": "  /rag                         show all RAG settings\n"
                        "  /rag topk <n>                results to retrieve (1–20)\n"
                        "  /rag threshold <0.0–1.0>     min similarity score\n"
                        "  /rag hybrid                  toggle hybrid BM25+vector\n"
                        "  /rag rerank                  toggle cross-encoder reranker\n"
                        "  /rag rerank-model <model>    set reranker model (HF ID or local path)\n"
                        "  /rag hyde                    toggle HyDE query expansion\n"
                        "  /rag multi                   toggle multi-query expansion\n"
                        "  /rag step-back               toggle step-back prompting\n"
                        "  /rag decompose               toggle query decomposition\n"
                        "  /rag compress                toggle LLM context compression\n"
                        "  /rag cite                    toggle inline [Document N] citations\n"
                        "  /rag raptor                  toggle RAPTOR hierarchical indexing\n"
                        "  /rag graph-rag               toggle GraphRAG entity retrieval",
                        "sessions": "  /sessions                    list recent saved sessions\n"
                        "  /resume <id>                 load a session by ID\n"
                        "  Sessions auto-save after each turn.",
                        "keys": "  /keys                        show API key status for all providers\n"
                        "  /keys set <provider>         interactively set an API key\n"
                        "  providers: gemini, openai, brave, ollama_cloud\n"
                        "  Keys are saved to ~/.axon/.env and loaded at startup.",
                        "project": "  /project                               show active project + list all\n"
                        "  /project list                          list all projects and mounted shares\n"
                        "  /project new <name>                    create a new project and switch to it\n"
                        "  /project new <name> <desc>             create with description\n"
                        "  /project new <parent>/<child>          create a sub-project (up to 5 levels)\n"
                        "  /project switch <name>                 switch to an existing local project\n"
                        "  /project switch <parent>/<child>       switch to a sub-project\n"
                        "  /project switch default                return to the global knowledge base\n"
                        "  /project switch @projects              merged view of all local projects\n"
                        "  /project switch @mounts                merged view of all mounted shares\n"
                        "  /project switch @store                 merged view of entire AxonStore\n"
                        "  /project switch mounts/<name>          switch to a mounted share\n"
                        "  /project delete <name>                 delete a leaf project and its data\n"
                        "  /project folder                        open the active project folder\n"
                        "\n"
                        "  Projects are stored in ~/.axon/projects/<name>/\n"
                        "  Sub-projects use nested subs/ directories (max depth: 5).\n"
                        "  Switching to a parent project shows merged data across all sub-projects.\n"
                        "  Use /ingest after switching to add documents to the current project.",
                        "share": "  /share list                              list all issued and received shares\n"
                        "  /share generate <project> <grantee>      generate a read-only share key\n"
                        "  /share redeem <share_string>              mount a shared project\n"
                        "  /share revoke <key_id>                   revoke a previously issued share\n"
                        "\n"
                        "  Requires AxonStore mode — run /store init first.\n"
                        "  Share strings are base64-encoded payloads; send them out-of-band.\n"
                        "  Mounted shares appear under mounts/ in your project list and can be\n"
                        "  switched to with /project switch mounts/<name>.",
                        "store": "  /store whoami                  show AxonStore identity and active project\n"
                        "  /store init <base_path>        initialise multi-user AxonStore at a path\n"
                        "\n"
                        "  Example: /store init ~/axon_data\n"
                        "  Creates: <base_path>/AxonStore/<username>/{default,projects,mounts,.shares}/\n"
                        "  Config is updated and persisted to ~/.config/axon/config.yaml.",
                        "graph": "  /graph status                  show entity count, edges, community summaries\n"
                        "  /graph finalize                trigger community detection rebuild\n"
                        "  /graph viz [path]              export graph as HTML (opens in browser)\n"
                        "\n"
                        "  GraphRAG must be enabled: /rag graph-rag\n"
                        "  Finalize is useful after batch ingest with community summarisation deferred.",
                        "refresh": "  /refresh                       re-ingest files whose content has changed\n"
                        "\n"
                        "  Computes current content hash for each tracked file and compares\n"
                        "  to the stored hash from the last ingest. Only changed files are re-ingested.\n"
                        "  Use /stale to preview which files are old before refreshing.",
                        "stale": "  /stale                         list documents older than 7 days\n"
                        "  /stale <days>                  list documents older than N days\n"
                        "\n"
                        "  Reports age based on the last ingest timestamp for each source.\n"
                        "  Use /refresh to re-ingest any changed files.",
                    }
                    key = arg.lstrip("/")
                    if key in _detail:
                        print(f"\n{_detail[key]}\n")
                    else:
                        print(f"  No detail for '{arg}'. Available: {', '.join(_detail)}")
                else:
                    print(
                        "\n"
                        "  /clear          clear knowledge base for current project\n"
                        "  /compact        summarise conversation to free context\n"
                        "  /context        show current conversation context size\n"
                        "  /discuss        toggle discussion fallback (general knowledge)\n"
                        "  /embed [model]  show or switch embedding model\n"
                        "  /graph [sub]    GraphRAG status, finalize communities, or viz export\n"
                        "  /help [cmd]     show this help or details for a command\n"
                        "  /ingest <path>  ingest a file, directory, or glob\n"
                        "  /keys           show/set API keys (gemini, openai, brave, ollama_cloud)\n"
                        "  /list           list ingested documents\n"
                        "  /llm [opt val]  show or set LLM settings (temperature)\n"
                        "  /model [model]  show or switch LLM model\n"
                        "  /project [sub]  manage projects (list, new, switch, delete, folder)\n"
                        "  /pull <name>    pull an Ollama model\n"
                        "  /quit           exit Axon\n"
                        "  /rag [opt val]  show or set retrieval settings (topk, threshold, hybrid, …)\n"
                        "  /refresh        re-ingest documents whose content has changed\n"
                        "  /resume <id>    load a saved session\n"
                        "  /retry          retry the last query\n"
                        "  /search         toggle Brave web search fallback\n"
                        "  /sessions       list recent saved sessions\n"
                        "  /share [sub]    share projects (generate, redeem, revoke, list)\n"
                        "  /stale [days]   list documents not refreshed in N days (default: 7)\n"
                        "  /store [sub]    AxonStore multi-user mode (init, whoami)\n"
                        "\n"
                        "  Shell:   !<cmd>  run a shell command (local-only default)\n"
                        "  Files:   @<file>  attach file context  ·  @<folder>/  attach all text files\n"
                        "\n"
                        "  /help <cmd>  for details  ·  e.g.  /help rag   /help share   /help project\n"
                        "  Tab  autocomplete  ·  ↑↓  history  ·  Ctrl+C  cancel  ·  Ctrl+D  exit\n"
                    )

            elif cmd == "/list":
                docs = brain.list_documents()
                if not docs:
                    print("  Knowledge base is empty.")
                else:
                    total = sum(d["chunks"] for d in docs)
                    print(f"\n  {len(docs)} file(s), {total} chunk(s)\n")
                    for d in docs:
                        print(f"  {d['source']:<60} {d['chunks']:>6}")
                    print()

            elif cmd == "/ingest":
                if not arg:
                    print("  Usage: /ingest <path|glob>  e.g. /ingest ./docs  /ingest ./src/*.py")
                else:
                    from axon.projects import ensure_project

                    # Prompt to create a project if none exist and currently in 'default'
                    if brain.should_recommend_project():
                        try:
                            print(
                                "\n  \033[1mNote\033[0m: You are about to ingest into the 'default' project."
                            )
                            print(
                                "  It is recommended to create a dedicated project to keep your data organized."
                            )
                            confirm = (
                                _read_input("  Create a new project now? [y/N]: ").strip().lower()
                            )
                            if confirm == "y":
                                new_name = _read_input("  New project name: ").strip().lower()
                                if new_name:
                                    try:
                                        ensure_project(new_name)
                                        brain.switch_project(new_name)
                                        print(f"  Switched to project '{new_name}'.\n")
                                    except ValueError as e:
                                        print(f"  {e}")
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Cancelled project check.")

                    import glob as _glob

                    from axon.loaders import DirectoryLoader

                    # Expand glob pattern; fallback to literal path
                    matched = sorted(_glob.glob(arg, recursive=True))
                    if not matched:
                        # No glob match — try as plain directory
                        if os.path.isdir(arg):
                            matched = [arg]
                        else:
                            print(f"  No files matched: {arg}")
                    if matched:
                        loader_mgr = DirectoryLoader()
                        ingested, skipped = 0, 0
                        for path in matched:
                            if os.path.isdir(path):
                                print(f"  {path} …", end="", flush=True)
                                asyncio.run(brain.load_directory(path))
                                print("  done")
                                ingested += 1
                            elif os.path.isfile(path):
                                ext = os.path.splitext(path)[1].lower()
                                if ext in loader_mgr.loaders:
                                    brain.ingest(loader_mgr.loaders[ext].load(path))
                                    print(f"  {path}")
                                    ingested += 1
                                else:
                                    print(f"  !  Skipped (unsupported type): {path}")
                                    skipped += 1
                        print(f"  Done — {ingested} ingested, {skipped} skipped.")

            elif cmd == "/model":
                _PROVIDERS = (
                    "ollama",
                    "gemini",
                    "openai",
                    "ollama_cloud",
                    "vllm",
                    "github_copilot",
                )
                if not arg:
                    print(f"  LLM:       {brain.config.llm_provider}/{brain.config.llm_model}")
                    print(
                        f"  Embedding: {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("  Usage:   /model <model>              (auto-detect provider)")
                    print("           /model <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_PROVIDERS)}")
                    print(
                        f"  vLLM URL:  {brain.config.vllm_base_url}  (change with /vllm-url <url>)"
                    )
                elif "/" in arg:
                    provider, model = arg.split("/", 1)
                    if provider not in _PROVIDERS:
                        print(
                            f"  Unknown provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}"
                        )
                    else:
                        brain.config.llm_provider = provider
                        brain.config.llm_model = model
                        brain.llm = OpenLLM(brain.config)
                        print(f"  Switched LLM to {provider}/{model}")
                        if provider == "vllm":
                            print(
                                f"  ℹ️  vLLM server: {brain.config.vllm_base_url}  (change with /vllm-url <url>)"
                            )
                        elif provider != "ollama":
                            _prompt_key_if_missing(provider, brain)
                else:
                    inferred = _infer_provider(arg)
                    brain.config.llm_provider = inferred
                    brain.config.llm_model = arg
                    brain.llm = OpenLLM(brain.config)
                    print(f"  Switched LLM to {inferred}/{arg}")
                    _prompt_key_if_missing(inferred, brain)

            elif cmd == "/vllm-url":
                if not arg:
                    print(f"  Current vLLM base URL: {brain.config.vllm_base_url}")
                    print("  Usage: /vllm-url http://host:port/v1")
                else:
                    brain.config.vllm_base_url = arg
                    brain.llm._openai_clients = {}  # invalidate cached client
                    print(f"  vLLM base URL set to {arg}")

            elif cmd == "/embed":
                _EMBED_PROVIDERS = ("sentence_transformers", "ollama", "fastembed", "openai")
                if not arg:
                    print(
                        f"  Current:   {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("  Usage:   /embed <model>              (keep current provider)")
                    print("           /embed <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_EMBED_PROVIDERS)}")
                    print("  Examples:")
                    print("    /embed all-MiniLM-L6-v2                    (sentence_transformers)")
                    print("    /embed /path/to/local/model                (local folder)")
                    print("    /embed ollama/nomic-embed-text")
                    print("    /embed fastembed/BAAI/bge-small-en")
                    print("  !  Changing embedding model invalidates existing indexed documents.")
                else:
                    if "/" in arg:
                        provider, model = arg.split("/", 1)
                        if provider not in _EMBED_PROVIDERS:
                            # Could be a path like /home/user/model — treat as local st path
                            provider = brain.config.embedding_provider
                            model = arg
                        else:
                            brain.config.embedding_provider = provider
                            brain.config.embedding_model = model
                    else:
                        provider = brain.config.embedding_provider
                        model = arg
                        brain.config.embedding_model = model
                    try:
                        print("  ⠿ Loading embedding model…", end="", flush=True)
                        brain.embedding = OpenEmbedding(brain.config)
                        print(
                            f"\r  Embedding switched to {brain.config.embedding_provider}/{brain.config.embedding_model}"
                        )
                        print("  Re-ingest your documents so they use the new embedding model.")
                    except Exception as e:
                        print(f"\r  Failed to load embedding: {e}")

            elif cmd == "/pull":
                if not arg:
                    print("  Usage: /pull <model-name>")
                else:
                    try:
                        import ollama as _ollama

                        print(f"  Pulling '{arg}' …")
                        last_status = ""
                        for chunk in _ollama.pull(arg, stream=True):
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
                                line = f"  {status}: {int(completed/total*100)}%"
                            elif status:
                                line = f"  {status}"
                            else:
                                continue
                            # Pad to clear previous longer line
                            print(f"\r{line:<60}", end="", flush=True)
                            last_status = line  # noqa: F841
                        print(f"\r  '{arg}' ready.{' ' * 50}")
                    except Exception as e:
                        print(f"  Pull failed: {e}")

            elif cmd == "/graph-viz":
                import hashlib as _hashlib
                import time as _time
                from pathlib import Path as _Path

                if arg.strip():
                    _out_path = arg.strip()
                else:
                    _ts = _time.strftime("%Y%m%dT%H%M%S")
                    _qhash = _hashlib.sha1(b"graph").hexdigest()[:8]
                    import os as _os

                    _axon_home = _Path(_os.environ.get("AXON_HOME", str(_Path.home() / ".axon")))
                    _snap_dir = _axon_home / "cache" / "graphs" / f"{_ts}_{_qhash}"
                    try:
                        _snap_dir.mkdir(parents=True, exist_ok=True)
                        _out_path = str(_snap_dir / "knowledge-graph.html")
                    except OSError:
                        import tempfile as _tf

                        _out_path = str(_Path(_tf.gettempdir()) / f"axon_graph_{_ts}_{_qhash}.html")
                try:
                    brain.export_graph_html(_out_path)
                    print(f"  Graph visualization saved → {_out_path}")
                    print("  Open in your browser to explore the entity–relation graph.")
                except ImportError as _e:
                    print(f"  {_e}")
                except Exception as _e:
                    print(f"  Failed to export graph: {_e}")

            elif cmd == "/clear":
                _confirm = (
                    _read_input("  Clear knowledge base for the current project? [y/N]: ")
                    .strip()
                    .lower()
                )
                if _confirm not in ("y", "yes"):
                    print("  Clear cancelled.")
                    continue
                try:
                    brain._assert_write_allowed("clear")
                    clear_active_project(brain)
                    from axon import api as _api

                    project_key = getattr(brain, "_active_project", "default")
                    _api._source_hashes.pop(project_key, None)
                    if project_key == "default":
                        _api._source_hashes.pop("_global", None)
                    print(f"  Knowledge base cleared for project '{brain._active_project}'.")
                except PermissionError as _e:
                    print(f"  {_e}")
                except Exception as _e:
                    print(f"  Clear failed: {_e}")

            elif cmd == "/search":
                if brain.config.offline_mode:
                    print("  Offline mode is ON — web search is disabled.")
                elif brain.config.truth_grounding:
                    brain.config.truth_grounding = False
                    print("  Web search OFF — answers from local knowledge only.")
                else:
                    if not brain.config.brave_api_key:
                        print("  BRAVE_API_KEY is not set. Export it and restart, or set it with:")
                        print("     export BRAVE_API_KEY=your_key")
                    else:
                        brain.config.truth_grounding = True
                        print(
                            "  Web search ON — Brave Search will be used as fallback when local knowledge is insufficient."
                        )

            elif cmd == "/discuss":
                brain.config.discussion_fallback = not brain.config.discussion_fallback
                state = "ON" if brain.config.discussion_fallback else "OFF"
                print(f"  Discussion mode {state}.")

            elif cmd == "/rag":
                if not arg:
                    _grag_mode = getattr(brain.config, "graph_rag_mode", "local")
                    print(
                        f"\n  top-k           · {brain.config.top_k}\n"
                        f"  threshold       · {brain.config.similarity_threshold}\n"
                        f"  hybrid          · {'ON' if brain.config.hybrid_search else 'OFF'}\n"
                        f"  rerank          · {'ON' if brain.config.rerank else 'OFF'}"
                        + (f"  [{brain.config.reranker_model}]" if brain.config.rerank else "")
                        + "\n"
                        f"  hyde            · {'ON' if brain.config.hyde else 'OFF'}\n"
                        f"  multi-query     · {'ON' if brain.config.multi_query else 'OFF'}\n"
                        f"  step-back       · {'ON' if brain.config.step_back else 'OFF'}\n"
                        f"  decompose       · {'ON' if brain.config.query_decompose else 'OFF'}\n"
                        f"  compress        · {'ON' if brain.config.compress_context else 'OFF'}\n"
                        f"  sentence-window · {'ON' if getattr(brain.config, 'sentence_window', False) else 'OFF'}\n"
                        f"  crag-lite       · {'ON' if getattr(brain.config, 'crag_lite', False) else 'OFF'}\n"
                        f"  code-graph      · {'ON' if getattr(brain.config, 'code_graph', False) else 'OFF'}\n"
                        f"  raptor          · {'ON' if brain.config.raptor else 'OFF'}\n"
                        f"  graph-rag       · {'ON' if brain.config.graph_rag else 'OFF'}\n"
                        f"  graph-rag-mode  · {_grag_mode}\n"
                        f"\n  /help rag   for usage details\n"
                    )
                else:
                    rag_parts = arg.split(maxsplit=1)
                    rag_opt = rag_parts[0].lower()
                    rag_val = rag_parts[1] if len(rag_parts) > 1 else ""
                    if rag_opt == "topk":
                        try:
                            n = int(rag_val)
                            assert 1 <= n <= 50
                            brain.config.top_k = n
                            print(f"  top-k set to {n}")
                        except Exception:
                            print("  Usage: /rag topk <integer 1–50>")
                    elif rag_opt == "threshold":
                        try:
                            v = float(rag_val)
                            assert 0.0 <= v <= 1.0
                            brain.config.similarity_threshold = v
                            print(f"  threshold set to {v}")
                        except Exception:
                            print("  Usage: /rag threshold <float 0.0–1.0>")
                    elif rag_opt == "hybrid":
                        brain.config.hybrid_search = not brain.config.hybrid_search
                        print(f"  Hybrid search {'ON' if brain.config.hybrid_search else 'OFF'}")
                    elif rag_opt == "rerank":
                        brain.config.rerank = not brain.config.rerank
                        print(f"  Reranker {'ON' if brain.config.rerank else 'OFF'}")
                    elif rag_opt == "hyde":
                        brain.config.hyde = not brain.config.hyde
                        print(f"  HyDE {'ON' if brain.config.hyde else 'OFF'}")
                    elif rag_opt == "multi":
                        brain.config.multi_query = not brain.config.multi_query
                        print(f"  Multi-query {'ON' if brain.config.multi_query else 'OFF'}")
                    elif rag_opt == "step-back":
                        brain.config.step_back = not brain.config.step_back
                        print(f"  Step-back prompting {'ON' if brain.config.step_back else 'OFF'}")
                    elif rag_opt == "decompose":
                        brain.config.query_decompose = not brain.config.query_decompose
                        print(
                            f"  Query decomposition {'ON' if brain.config.query_decompose else 'OFF'}"
                        )
                    elif rag_opt == "compress":
                        brain.config.compress_context = not brain.config.compress_context
                        print(
                            f"  Context compression {'ON' if brain.config.compress_context else 'OFF'}"
                        )
                    elif rag_opt == "cite":
                        brain.config.cite = not brain.config.cite
                        print(f"  Inline citations {'ON' if brain.config.cite else 'OFF'}")
                    elif rag_opt == "raptor":
                        brain.config.raptor = not brain.config.raptor
                        print(
                            f"  RAPTOR hierarchical indexing {'ON' if brain.config.raptor else 'OFF'}"
                        )
                    elif rag_opt in ("graph-rag", "graph_rag", "graphrag"):
                        brain.config.graph_rag = not brain.config.graph_rag
                        print(
                            f"  GraphRAG entity retrieval {'ON' if brain.config.graph_rag else 'OFF'}"
                        )
                    elif rag_opt in ("sentence-window", "sentence_window"):
                        _on = (
                            rag_val.lower() in ("on", "true", "1", "")
                            if rag_val
                            else not getattr(brain.config, "sentence_window", False)
                        )
                        if rag_val and rag_val.lower() not in (
                            "on",
                            "off",
                            "true",
                            "false",
                            "1",
                            "0",
                        ):
                            print("  Usage: /rag sentence-window on|off")
                        else:
                            _on = (
                                rag_val.lower() in ("on", "true", "1")
                                if rag_val
                                else not getattr(brain.config, "sentence_window", False)
                            )
                            brain.config.sentence_window = _on
                            print(f"  Sentence-window retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("sentence-window-size", "sentence_window_size"):
                        try:
                            _sz = int(rag_val)
                            assert 1 <= _sz <= 10
                            brain.config.sentence_window_size = _sz
                            print(f"  Sentence-window size set to {_sz}")
                        except Exception:
                            print("  Usage: /rag sentence-window-size <integer 1–10>")
                    elif rag_opt in ("crag-lite", "crag_lite"):
                        _on = (
                            rag_val.lower() in ("on", "true", "1")
                            if rag_val
                            else not getattr(brain.config, "crag_lite", False)
                        )
                        brain.config.crag_lite = _on
                        print(f"  CRAG-lite corrective retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("code-graph", "code_graph"):
                        _on = (
                            rag_val.lower() in ("on", "true", "1")
                            if rag_val
                            else not getattr(brain.config, "code_graph", False)
                        )
                        brain.config.code_graph = _on
                        print(f"  Code-graph retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("graph-rag-mode", "graph_rag_mode"):
                        _valid_modes = ("local", "global", "hybrid", "auto")
                        if rag_val.lower() not in _valid_modes:
                            print("  Usage: /rag graph-rag-mode local|global|hybrid|auto")
                        else:
                            brain.config.graph_rag_mode = rag_val.lower()
                            print(f"  GraphRAG mode set to '{rag_val.lower()}'")
                    elif rag_opt == "rerank-model":
                        if not rag_val:
                            print(f"  Current reranker: {brain.config.reranker_model}")
                            print("  Usage: /rag rerank-model <HuggingFace ID or local path>")
                            print("  e.g.  /rag rerank-model BAAI/bge-reranker-base")
                            print("        /rag rerank-model ./models/bge-reranker-base")
                        else:
                            resolved = brain._resolve_model_path(rag_val)
                            if resolved != rag_val:
                                print(f"  Resolved to local path: {resolved}")
                            brain.config.reranker_model = resolved
                            brain.config.rerank = True  # auto-enable when setting a model
                            print(f"  Loading reranker '{resolved}'…")
                            try:
                                brain.reranker = OpenReranker(brain.config)
                                print(f"  Reranker → {resolved}  (rerank: ON)")
                            except Exception as e:
                                brain.config.rerank = False
                                print(f"  Failed to load reranker: {e}")
                    else:
                        print(
                            f"  Unknown option '{rag_opt}'. Try: topk, threshold, hybrid, rerank, rerank-model, "
                            f"hyde, multi, step-back, decompose, compress, cite, raptor, graph-rag, "
                            f"sentence-window, sentence-window-size, crag-lite, code-graph, graph-rag-mode"
                        )

            elif cmd == "/llm":
                if not arg:
                    print(
                        f"\n  temperature  · {brain.config.llm_temperature}\n"
                        f"  provider     · {brain.config.llm_provider}\n"
                        f"  model        · {brain.config.llm_model}\n"
                        f"\n  /llm temperature <0.0–2.0>   set generation temperature\n"
                    )
                else:
                    llm_parts = arg.split(maxsplit=1)
                    llm_opt = llm_parts[0].lower()
                    llm_val = llm_parts[1] if len(llm_parts) > 1 else ""
                    if llm_opt == "temperature":
                        try:
                            v = float(llm_val)
                            assert 0.0 <= v <= 2.0
                            brain.config.llm_temperature = v
                            print(f"  Temperature set to {v}")
                        except Exception:
                            print("  Usage: /llm temperature <float 0.0–2.0>")
                    else:
                        print(f"  Unknown option '{llm_opt}'. Available: temperature")

            elif cmd == "/compact":
                _do_compact(brain, chat_history)
                _save_session(session)

            elif cmd == "/project":
                from axon.projects import (
                    ProjectHasChildrenError,
                    delete_project,
                    ensure_project,
                    list_projects,
                    project_dir,
                )

                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if not sub or sub == "list":
                    projects = list_projects()
                    active = brain._active_project

                    if not projects:
                        print("  No projects yet. Use /project new <name> to create one.")
                    else:
                        print()
                        _print_project_tree(projects, active)

                    # Show mounted shares if AxonStore mode is active
                    try:
                        from axon.projects import list_share_mounts as _list_mounts

                        _user_dir = Path(brain.config.projects_root)
                        _mounts = _list_mounts(_user_dir)
                        if _mounts:
                            print("\n  Mounted shares:")
                            for _m in _mounts:
                                _broken = "  [broken]" if _m.get("is_broken") else ""
                                print(f"    mounts/{_m['name']}  (owner: {_m['owner']}){_broken}")
                    except Exception:
                        pass

                    print(f"\n  Active: {active}")
                    print("  /project new <name>                      create + switch")
                    print("  /project new <parent>/<name>             create sub-project")
                    print("  /project switch <name>                   switch to existing")
                    print("  /project switch @projects|@mounts|@store switch to merged scope")
                    print("  /project switch mounts/<name>            switch to mounted share")
                    print("  /project folder                          open active project folder\n")

                elif sub == "new":
                    if not sub_arg:
                        print("  Usage: /project new <name>  [description]")
                        print("         /project new research/papers  (sub-project)")
                    else:
                        name_parts = sub_arg.split(maxsplit=1)
                        proj_name = name_parts[0].lower()
                        proj_desc = name_parts[1] if len(name_parts) > 1 else ""
                        try:
                            ensure_project(proj_name, proj_desc)
                            brain.switch_project(proj_name)
                            print(f"  Created and switched to project '{proj_name}'")
                            print(f"  {project_dir(proj_name)}")
                            print("  Use /ingest to add documents to this project.\n")
                        except ValueError as e:
                            print(f"  {e}")

                elif sub == "switch":
                    if not sub_arg:
                        print("  Usage: /project switch <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        # Allow merged scopes (@projects, @mounts, @store), mounted scopes
                        # (mounts/<name>), and default — delegate directly to backend.
                        _is_special = (
                            proj_name == "default"
                            or proj_name.startswith("@")
                            or proj_name.startswith("mounts/")
                        )
                        if _is_special or project_dir(proj_name).exists():
                            try:
                                brain.switch_project(proj_name)
                                is_merged = isinstance(brain.vector_store, MultiVectorStore)
                                if is_merged:
                                    print(f"  Switched to project '{proj_name}'  [merged view]\n")
                                elif brain.vector_store.provider == "chroma":
                                    count = brain.vector_store.collection.count()
                                    print(
                                        f"  Switched to project '{proj_name}'  ({count} chunks)\n"
                                    )
                                else:
                                    print(f"  Switched to project '{proj_name}'\n")
                            except Exception as e:
                                print(f"  {e}")
                        else:
                            print(
                                f"  Project '{proj_name}' not found. Use /project list or /project new {proj_name}"
                            )

                elif sub == "delete":
                    if not sub_arg:
                        print("  Usage: /project delete <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        try:
                            confirm = (
                                _read_input(
                                    f"  !  Delete project '{proj_name}' and ALL its data? [y/N]: "
                                )
                                .strip()
                                .lower()
                            )
                        except (EOFError, KeyboardInterrupt):
                            confirm = "n"
                        if confirm == "y":
                            try:
                                if brain._active_project == proj_name:
                                    brain.switch_project("default")
                                    print("  ↩️  Switched back to default project.")
                                delete_project(proj_name)
                                print(f"  Deleted project '{proj_name}'.\n")
                            except ProjectHasChildrenError as e:
                                print(f"  {e}")
                            except ValueError as e:
                                print(f"  {e}")
                        else:
                            print("  Cancelled.")

                elif sub == "folder":
                    active = brain._active_project
                    if active == "default":
                        print("  Default project uses config paths:")
                        print(f"    Vector store: {brain.config.vector_store_path}")
                        print(f"    BM25 index:   {brain.config.bm25_path}\n")
                    else:
                        folder = str(project_dir(active))
                        print(f"  {folder}")
                        import subprocess

                        try:
                            if os.name == "nt":
                                subprocess.Popen(["explorer", folder])
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", folder])
                            else:
                                subprocess.Popen(["xdg-open", folder])
                        except Exception:
                            pass

                else:
                    print(f"  Unknown sub-command '{sub}'. Try: list, new, switch, delete, folder")

            elif cmd == "/retry":
                if not _last_query:
                    print("  Nothing to retry — no previous query.")
                else:
                    user_input = _last_query
                    print(f"  ↩️  Retrying: {user_input}")

            elif cmd == "/context":
                _show_context(brain, chat_history, _last_sources, _last_query)

            elif cmd == "/sessions":
                _print_sessions(_list_sessions(project=brain._active_project))

            elif cmd == "/resume":
                if not arg:
                    print("  Usage: /resume <session-id>")
                else:
                    loaded = _load_session(arg, project=brain._active_project)
                    if loaded is None:
                        print(f"  Session '{arg}' not found. Use /sessions to list.")
                    else:
                        session = loaded
                        chat_history.clear()
                        chat_history.extend(session["history"])
                        turns = len(chat_history) // 2
                        print(f"  Loaded session {session['id']}  ({turns} turns)\n")

            elif cmd == "/keys":
                _env_file = Path.home() / ".axon" / ".env"
                _provider_keys = {
                    "gemini": "GEMINI_API_KEY",
                    "openai": "OPENAI_API_KEY",
                    "brave": "BRAVE_API_KEY",
                    "ollama_cloud": "OLLAMA_CLOUD_KEY",
                    "github_copilot": "GITHUB_COPILOT_PAT",
                }
                if arg.lower().startswith("set"):
                    set_parts = arg.split(maxsplit=1)
                    prov = set_parts[1].lower().strip() if len(set_parts) > 1 else ""
                    if not prov or prov not in _provider_keys:
                        print("  Usage: /keys set <provider>")
                        print(f"  Providers: {', '.join(_provider_keys)}")
                    elif prov == "github_copilot":
                        print("  Starting GitHub OAuth device flow…")
                        try:
                            new_key = _copilot_device_flow()
                        except (EOFError, KeyboardInterrupt, RuntimeError) as e:
                            print(f"\n  Cancelled: {e}")
                        else:
                            env_name = _provider_keys[prov]
                            _save_env_key(env_name, new_key)
                            brain.config.copilot_pat = new_key
                            for k in ("_copilot", "_copilot_session", "_copilot_token"):
                                brain.llm._openai_clients.pop(k, None)
                            print(f"  {env_name} saved to {_env_file} and applied.")
                            print("  Switch provider: /model github_copilot/<model>")
                    else:
                        env_name = _provider_keys[prov]
                        try:
                            import getpass

                            new_key = getpass.getpass(f"  Enter {env_name} (hidden): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Cancelled.")
                        else:
                            if new_key:
                                _save_env_key(env_name, new_key)
                                if prov == "brave":
                                    brain.config.brave_api_key = new_key
                                elif prov == "gemini":
                                    brain.config.gemini_api_key = new_key
                                elif prov == "openai":
                                    brain.config.api_key = new_key
                                elif prov == "ollama_cloud":
                                    brain.config.ollama_cloud_key = new_key
                                print(f"  {env_name} saved to {_env_file} and applied.")
                                print(f"  Switch provider: /model {prov}/<model-name>")
                            else:
                                print("  No key entered — nothing saved.")
                else:
                    print("\n  API Key Status\n  " + "─" * 50)
                    for prov, env_name in _provider_keys.items():
                        val = os.environ.get(env_name, "")
                        if val:
                            masked = val[:4] + "****" + val[-2:] if len(val) > 6 else "****"
                            status = f"set ({masked})"
                        else:
                            status = "not set"
                        print(f"  {prov:<14} {env_name:<22} {status}")
                    if _env_file.exists():
                        print(f"\n  Keys file: {_env_file}")
                    else:
                        print("\n  No keys file yet. Use /keys set <provider> to add keys.")
                    print("  /keys set <provider>  to set a key interactively")
                    print("  /help keys            for provider URLs and usage\n")

            elif cmd == "/share":
                # ── /share — project sharing lifecycle ──────────────────────────
                from axon import shares as _shares_mod

                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if not brain.config.axon_store_mode:
                    print("  AxonStore mode is not active. Run /store init <path> first.")
                elif not sub or sub == "list":
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
                            print(f"    {s['owner']}/{s['project']} mounted as {s['mount']}  [ro]")
                    else:
                        print("    (none)")
                    print()

                elif sub == "generate":
                    # Usage: /share generate <project> <grantee>
                    parts = sub_arg.split()
                    if len(parts) < 2:
                        print("  Usage: /share generate <project> <grantee>")
                    else:
                        proj = parts[0]
                        grantee = parts[1]
                        user_dir = Path(brain.config.projects_root)
                        # Resolve nested projects via subs/ layout (e.g. research/papers → research/subs/papers)
                        _segs = proj.split("/")
                        proj_dir = user_dir / _segs[0]
                        for _seg in _segs[1:]:
                            proj_dir = proj_dir / "subs" / _seg
                        if not proj_dir.exists() or not (proj_dir / "meta.json").exists():
                            print(
                                f"  Project '{proj}' not found. Use /project list to see projects."
                            )
                        else:
                            try:
                                result = _shares_mod.generate_share_key(
                                    owner_user_dir=user_dir,
                                    project=proj,
                                    grantee=grantee,
                                )
                                print(f"\n  Share key generated for project '{proj}'")
                                print(f"  Grantee:      {grantee}")
                                print("  Access:       read-only")
                                print(f"  Key ID:       {result['key_id']}")
                                print(f"\n  Share string (send this to {grantee}):")
                                print(f"\n    {result['share_string']}\n")
                                print(f"  Revoke with:  /share revoke {result['key_id']}\n")
                            except Exception as e:
                                print(f"  Share generation failed: {e}")

                elif sub == "redeem":
                    # Usage: /share redeem <share_string>
                    if not sub_arg:
                        print("  Usage: /share redeem <share_string>")
                    else:
                        user_dir = Path(brain.config.projects_root)
                        try:
                            result = _shares_mod.redeem_share_key(
                                grantee_user_dir=user_dir,
                                share_string=sub_arg.strip(),
                            )
                            print("\n  Share redeemed!")
                            print(f"  Project '{result['project']}' from {result['owner']}")
                            print(
                                f"  Mounted at:  mounts/{result.get('mount_name', result['owner'] + '_' + result['project'])}"
                            )
                            print("  Access:      read-only\n")
                        except (ValueError, NotImplementedError) as e:
                            print(f"  Redeem failed: {e}")
                        except Exception as e:
                            print(f"  Redeem failed: {e}")

                elif sub == "revoke":
                    # Usage: /share revoke <key_id>
                    if not sub_arg:
                        print("  Usage: /share revoke <key_id>")
                    else:
                        user_dir = Path(brain.config.projects_root)
                        try:
                            result = _shares_mod.revoke_share_key(
                                owner_user_dir=user_dir,
                                key_id=sub_arg.strip(),
                            )
                            print(f"  Share '{result['key_id']}' revoked.")
                        except ValueError as e:
                            print(f"  Revoke failed: {e}")
                        except Exception as e:
                            print(f"  Revoke failed: {e}")

                else:
                    print(f"  Unknown sub-command '{sub}'.")
                    print(
                        "  Usage: /share list | generate <project> <grantee> [--write] | redeem <string> | revoke <key_id>"
                    )

            elif cmd == "/store":
                # ── /store — AxonStore initialisation + identity ─────────────────
                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if not sub or sub == "whoami":
                    import getpass as _gp

                    username = _gp.getuser()
                    if brain.config.axon_store_mode:
                        print("\n  AxonStore  active")
                        print(f"  User:       {username}")
                        print(f"  User dir:   {brain.config.projects_root}")
                        store_path = str(Path(brain.config.projects_root).parent)
                        print(f"  Store path: {store_path}")
                        print(f"  Project:    {brain._active_project}\n")
                    else:
                        print("\n  AxonStore  not active")
                        print(f"  User:       {username}")
                        print("  Run /store init <path> to enable multi-user mode.\n")

                elif sub == "init":
                    if not sub_arg:
                        print("  Usage: /store init <base_path>")
                        print("  Example: /store init ~/axon_data")
                    else:
                        import getpass as _gp

                        from axon.projects import ensure_user_project

                        base = Path(sub_arg.strip()).expanduser().resolve()
                        username = _gp.getuser()
                        store_root = base / "AxonStore"
                        user_dir = store_root / username
                        try:
                            ensure_user_project(user_dir)
                            brain.config.axon_store_base = str(base)
                            brain.config.axon_store_mode = True
                            brain.config.projects_root = str(user_dir)
                            brain.config.vector_store_path = str(
                                user_dir / "default" / "chroma_data"
                            )
                            brain.config.bm25_path = str(user_dir / "default" / "bm25_index")
                            try:
                                brain.config.save()
                            except Exception as _save_exc:
                                print(f"  Warning: could not save config: {_save_exc}")
                            print(f"\n  AxonStore initialised at {store_root}")
                            print(f"  Your directory:  {user_dir}")
                            print(f"  Username:        {username}")
                            print("  Use /share generate to share projects with others.\n")
                        except Exception as e:
                            print(f"  Store init failed: {e}")

                else:
                    print(f"  Unknown sub-command '{sub}'.")
                    print("  Usage: /store whoami | /store init <base_path>")

            elif cmd == "/refresh":
                # ── /refresh — re-ingest changed documents ───────────────────────
                import hashlib as _hl_r

                from axon.loaders import DirectoryLoader as _DL

                versions = brain.get_doc_versions()
                if not versions:
                    print("  No tracked documents. Use /ingest to add documents.")
                else:
                    _dl = _DL()
                    reingested, skipped, missing, errors = [], [], [], []
                    for source_path, record in versions.items():
                        if not os.path.exists(source_path):
                            missing.append(source_path)
                            continue
                        suffix = os.path.splitext(source_path)[1].lower()
                        loader_inst = _dl.loaders.get(suffix)
                        if loader_inst is None:
                            errors.append(f"{source_path}: no loader for extension '{suffix}'")
                            continue
                        try:
                            docs = loader_inst.load(source_path)
                            if not docs:
                                errors.append(f"{source_path}: loader returned no documents")
                                continue
                            combined = "".join(d.get("text", "") for d in docs)
                            current_hash = _hl_r.md5(
                                combined.encode("utf-8", errors="replace")
                            ).hexdigest()
                            if current_hash == record.get("content_hash"):
                                skipped.append(source_path)
                            else:
                                brain.ingest(docs)
                                reingested.append(source_path)
                        except Exception as _e:
                            errors.append(f"{source_path}: {_e}")
                    print("\n  Refresh complete:")
                    print(f"    Re-ingested: {len(reingested)}")
                    print(f"    Unchanged:   {len(skipped)}")
                    print(f"    Missing:     {len(missing)}")
                    if errors:
                        print(f"    Errors:      {len(errors)}")
                        for err in errors:
                            print(f"      {err}")
                    if reingested:
                        print("  Updated:")
                        for s in reingested:
                            print(f"    {s}")
                    print()

            elif cmd == "/stale":
                # ── /stale [days] — list documents not refreshed in N days ───────
                from datetime import datetime, timezone

                try:
                    threshold_days = int(arg) if arg.strip() else 7
                except ValueError:
                    print("  Usage: /stale [days]  (default: 7)")
                    threshold_days = -1

                if threshold_days >= 0:
                    cutoff = datetime.now(timezone.utc).timestamp() - threshold_days * 86400
                    versions = brain.get_doc_versions()
                    stale = []
                    for src, record in versions.items():
                        ts_str = record.get("ingested_at") or record.get("last_ingested_at")
                        if not ts_str:
                            continue
                        try:
                            ts = (
                                datetime.fromisoformat(ts_str.rstrip("Z"))
                                .replace(tzinfo=timezone.utc)
                                .timestamp()
                            )
                        except ValueError:
                            continue
                        if ts < cutoff:
                            age_days = round(
                                (datetime.now(timezone.utc).timestamp() - ts) / 86400, 1
                            )
                            stale.append((age_days, src, ts_str))
                    stale.sort(reverse=True)
                    if stale:
                        print(f"\n  Stale documents (>{threshold_days} days):")
                        for age, src, _ts in stale:
                            print(f"    {age:6.1f}d  {src}")
                        print(f"\n  Total: {len(stale)}")
                        print("  Run /refresh to re-ingest changed documents.\n")
                    else:
                        print(f"  All documents are fresh (threshold: {threshold_days} days).")

            elif cmd == "/graph":
                # ── /graph — GraphRAG status + finalize ──────────────────────────
                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""

                if not sub or sub == "status":
                    in_progress = getattr(brain, "_community_build_in_progress", False)
                    summaries = getattr(brain, "_community_summaries", {}) or {}
                    entities = getattr(brain, "_entity_graph", {})
                    relations = getattr(brain, "_relation_graph", {})
                    relation_edges = sum(len(v) for v in relations.values())
                    print("\n  GraphRAG status:")
                    print(f"    Entities:              {len(entities)}")
                    print(f"    Relation edges:        {relation_edges}")
                    print(f"    Community summaries:   {len(summaries)}")
                    print(f"    Community build:       {'in progress' if in_progress else 'idle'}")
                    print(f"    graph_rag enabled:     {brain.config.graph_rag}")
                    print()

                elif sub == "finalize":
                    if not brain.config.graph_rag:
                        print("  GraphRAG is disabled. Enable with /rag graph-rag first.")
                    else:
                        print("  Finalizing graph communities… (this may take a moment)")
                        try:
                            brain.finalize_graph()
                            summaries = getattr(brain, "_community_summaries", {}) or {}
                            print(f"  Done. {len(summaries)} community summaries generated.\n")
                        except Exception as e:
                            print(f"  Finalize failed: {e}")

                elif sub == "viz":
                    import hashlib as _hashlib_g
                    import time as _time_g

                    sub_parts2 = arg.split(maxsplit=1)
                    out_path = sub_parts2[1] if len(sub_parts2) > 1 else ""
                    try:
                        html = brain.export_graph_html(open_browser=False)
                        if out_path:
                            out = Path(out_path).expanduser()
                            out.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            _ts_g = _time_g.strftime("%Y%m%dT%H%M%S")
                            _qh_g = _hashlib_g.sha1(b"graph").hexdigest()[:8]
                            _preferred = (
                                Path(os.environ.get("AXON_HOME", str(Path.home() / ".axon")))
                                / "cache"
                                / "graphs"
                                / f"{_ts_g}_{_qh_g}"
                            )
                            try:
                                _preferred.mkdir(parents=True, exist_ok=True)
                                out = _preferred / "knowledge-graph.html"
                            except OSError:
                                import tempfile as _tf_g

                                out = Path(_tf_g.gettempdir()) / f"axon_graph_{_ts_g}_{_qh_g}.html"
                        out.write_text(html, encoding="utf-8")
                        print(f"  Graph saved to: {out}")
                        try:
                            import subprocess as _sp_g

                            if os.name == "nt":
                                _sp_g.Popen(["start", str(out)], shell=True)
                            elif sys.platform == "darwin":
                                _sp_g.Popen(["open", str(out)])
                            else:
                                _sp_g.Popen(["xdg-open", str(out)])
                        except Exception:
                            print(
                                "  Open the file in your browser to explore the entity–relation graph."
                            )
                    except Exception as e:
                        print(f"  Graph visualisation failed: {e}")

                else:
                    print(f"  Unknown sub-command '{sub}'.")
                    print("  Usage: /graph status | /graph finalize | /graph viz [path]")

            else:
                print(f"  Unknown command: {cmd}. Type /help for options.")

            if cmd != "/retry":
                continue

        # --- @file expansion: replace @path references with file contents ---
        query_text = _expand_at_files(user_input)
        if query_text != user_input:
            at_files = re.findall(r"@(\S+)", user_input)
            print(f"  Attached: {', '.join(at_files)}")

        # --- Regular query — use Rich Live for spinner + streaming response ---
        response_parts: list = []
        _cancelled = False
        try:
            from rich.console import Console as _RC
            from rich.live import Live as _RL
            from rich.markdown import Markdown as _RM
            from rich.text import Text as _RT

            _console = _RC()

            # ── Spinner phase (transient=True removes it cleanly when stopped) ──
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx = [0]

            def _spin_update(live: _RL) -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    live.update(_RT.from_markup(f"[bold yellow]Axon:[/bold yellow] {f} thinking…"))
                    _spin_idx[0] += 1

            if stream:
                token_gen = brain.query_stream(query_text, chat_history=chat_history)

                if not quiet:
                    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
                    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
                    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
                    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
                    tk = f"top-k:{brain.config.top_k}"
                    thr = f"thr:{brain.config.similarity_threshold}"
                    _snap = (m, s_v, d_v, h_v, tk, thr)

                    if _snap != _last_config_snapshot:
                        # Only show the parts that changed
                        changes = []
                        for i, val in enumerate(_snap):
                            if i >= len(_last_config_snapshot) or val != _last_config_snapshot[i]:
                                changes.append(val)
                        if changes:
                            print(f"\033[2m  {'  │  '.join(changes)}\033[0m")
                        _last_config_snapshot = _snap

                    print()
                    # Spinner until first real token arrives
                    with _RL(
                        _RT.from_markup("[bold yellow]Axon:[/bold yellow] ⠋ thinking…"),
                        console=_console,
                        transient=True,
                        refresh_per_second=10,
                    ) as _spin_live:
                        _st = threading.Thread(target=_spin_update, args=(_spin_live,), daemon=True)
                        _st.start()
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            response_parts.append(chunk)
                            break  # first token → exit spinner
                        _spin_stop.set()
                        _st.join(timeout=0.3)
                    # spinner gone (transient); cursor is at a clean line

                # Stream remaining tokens via Rich Live (plain text + cursor),
                # then swap to full Markdown on completion — no raw cursor
                # save/restore so the terminal scrollback is never corrupted.
                try:
                    _console.print("[bold yellow]Axon:[/bold yellow]")
                    _accumulated = "".join(response_parts)
                    with _RL(
                        _RT(_accumulated + " ▋"),
                        console=_console,
                        transient=False,
                        refresh_per_second=15,
                    ) as live:
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            _accumulated += chunk
                            response_parts.append(chunk)
                            live.update(_RT(_accumulated + " ▋"))
                        # All tokens received — swap plain text for Markdown
                        live.update(_RM(_accumulated))
                    print()
                except KeyboardInterrupt:
                    _cancelled = True
                    if _accumulated:
                        _console.print(_RM(_accumulated))
                    print("\n  !  Cancelled.\n")
            else:
                # Non-streaming: spinner while brain.query() blocks
                _spin_stop2 = threading.Event()
                _result: list = []
                _err: list = []

                if not quiet:
                    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
                    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
                    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
                    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
                    tk = f"top-k:{brain.config.top_k}"
                    thr = f"thr:{brain.config.similarity_threshold}"
                    _snap = (m, s_v, d_v, h_v, tk, thr)

                    if _snap != _last_config_snapshot:
                        # Only show the parts that changed
                        changes = []
                        for i, val in enumerate(_snap):
                            if i >= len(_last_config_snapshot) or val != _last_config_snapshot[i]:
                                changes.append(val)
                        if changes:
                            print(f"\033[2m  {'  │  '.join(changes)}\033[0m")
                        _last_config_snapshot = _snap

                def _run_query() -> None:
                    try:
                        _result.append(brain.query(query_text, chat_history=chat_history))
                    except Exception as exc:
                        _err.append(exc)
                    finally:
                        _spin_stop2.set()

                _qt = threading.Thread(target=_run_query, daemon=True)
                _qt.start()

                if not quiet:
                    print()
                    with _RL(
                        _RT.from_markup("[bold yellow]Axon:[/bold yellow] ⠋ thinking…"),
                        console=_console,
                        transient=True,
                        refresh_per_second=10,
                    ) as _spin_live2:
                        _st2 = threading.Thread(
                            target=_spin_update, args=(_spin_live2,), daemon=True
                        )
                        _st2.start()
                        _spin_stop2.wait()
                        _spin_stop.set()
                        _st2.join(timeout=0.3)
                else:
                    _qt.join()

                if _err:
                    raise _err[0]
                response = _result[0] if _result else ""
                print()  # blank line between You: and Axon:
                _console.print("[bold yellow]Axon:[/bold yellow]")
                _console.print(_RM(response))
                print()  # blank line after Brain response, before next You:
                response_parts = [response]

        except ImportError:
            # rich not available — plain fallback
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx = [0]

            def _spin_plain() -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    sys.stdout.write(f"\r  Axon: {f} thinking…")
                    sys.stdout.flush()
                    _spin_idx[0] += 1

            if not quiet:
                print()
                _spt = threading.Thread(target=_spin_plain, daemon=True)
                _spt.start()
            response = brain.query(query_text, chat_history=chat_history)
            if not quiet:
                _spin_stop.set()
            print(f"\n\033[1;33mAxon:\033[0m {response}\n")
            response_parts = [response]

        response = "".join(response_parts)
        if not _cancelled:
            # Append both turns so future queries have full context
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            _last_query = user_input
            _save_session(session)  # persist after every turn
