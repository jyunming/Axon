"""REPL, display helpers, slash-command UI, and @file expansion for Axon."""


from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

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
    "/agent",
    "/clear",
    "/compact",
    "/config ",
    "/context",
    "/debug",
    "/discuss",
    "/embed ",
    "/exit",
    "/governance ",
    "/graph ",
    "/graph-viz",
    "/help",
    "/ingest ",
    "/keys",
    "/list",
    "/llm ",
    "/model ",
    "/mount-refresh",
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
    "/theme ",
    "/vllm-url ",
]

_SLASH_CMD_DESC: dict[str, str] = {
    "/agent": "Toggle agent mode (LLM calls Axon tools)",
    "/clear": "Clear conversation history",
    "/compact": "Summarize and compact chat history",
    "/config": "Show current configuration",
    "/context": "Show or clear attached context files",
    "/debug": "Toggle verbose debug logging on/off",
    "/discuss": "Toggle discussion fallback mode",
    "/embed": "Switch embedding model",
    "/exit": "Exit Axon",
    "/governance": "Operator console (overview / audit / sessions / projects / graph-rebuild)",
    "/graph": "GraphRAG operations (build / query / export)",
    "/graph-viz": "Open graph visualisation in browser",
    "/help": "Show all commands",
    "/ingest": "Ingest a file or directory into the knowledge base",
    "/keys": "Show keyboard shortcuts",
    "/list": "List indexed documents",
    "/llm": "Adjust LLM parameters (e.g. temperature)",
    "/model": "Switch LLM model",
    "/mount-refresh": "Refresh a sealed mount from the owner's latest version",
    "/project": "Switch or manage project namespaces",
    "/pull": "Fetch and ingest from a URL",
    "/quit": "Exit Axon",
    "/rag": "Toggle RAG flags (hybrid / rerank / hyde / graph)",
    "/refresh": "Re-ingest files that have changed",
    "/resume": "Resume a previous session",
    "/retry": "Retry the last query",
    "/search": "Toggle semantic search mode",
    "/sessions": "List saved sessions",
    "/share": "Share a project or manage share keys",
    "/stale": "Show documents not refreshed recently",
    "/store": "AxonStore management (init / status / share)",
    "/theme": "Switch syntax-highlighting theme",
    "/vllm-url": "Set the vLLM server base URL",
}


_BLOCK_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\$([^$\n]{1,300}?)\$")

# ---------------------------------------------------------------------------
# Markdown code-block syntax-highlight theme (switchable via /theme)
# ---------------------------------------------------------------------------

_PREFS_FILE = Path.home() / ".axon" / "prefs.json"


def _load_prefs() -> dict:
    try:
        import json

        return json.loads(_PREFS_FILE.read_text())
    except Exception:
        return {}


def _save_pref(key: str, value: object) -> None:
    import json

    prefs = _load_prefs()
    prefs[key] = value
    _PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PREFS_FILE.write_text(json.dumps(prefs, indent=2))


_MD_CODE_THEME: str = str(_load_prefs().get("md_code_theme", "monokai"))

# ---------------------------------------------------------------------------
# GitHub-style callout pre-processor
# ---------------------------------------------------------------------------

_CALLOUT_RE = re.compile(
    r"^> \[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][^\n]*\n((?:^> [^\n]*\n?)*)",
    re.MULTILINE,
)
_CALLOUT_RICH: dict[str, tuple[str, str]] = {
    "NOTE": ("ℹ", "cyan"),
    "TIP": ("✦", "green"),
    "IMPORTANT": ("★", "magenta"),
    "WARNING": ("⚠", "yellow"),
    "CAUTION": ("✖", "red"),
}

_TASK_LIST_RE = re.compile(r"^([ \t]*[-*+] )\[([ xX])\] ", re.MULTILINE)
_STRIKETHROUGH_RE = re.compile(r"~~(.+?)~~", re.DOTALL)


def _fence_unfenced_code(text: str) -> str:
    """Wrap runs of code-like lines not already inside ``` fences.
    Detects consecutive lines that look like source code and wraps them in
    triple-backtick fences with auto-detected language tags.
    """
    if not text:
        return text
    _LANG_SIGNALS: list[tuple[str, re.Pattern]] = [
        ("python", re.compile(r"^(def |class |import |from |if __name__|@|#!/|async def )")),
        ("javascript", re.compile(r"^(function |const |let |var |export |module\.)")),
        ("typescript", re.compile(r"^(interface |type |export (type|interface|class|function))")),
        ("rust", re.compile(r"^(fn |pub |use |impl |struct |enum |mod |let mut )")),
        ("go", re.compile(r"^(package |func |import \(|var |type .+ struct)")),
        (
            "sql",
            re.compile(r"^(SELECT |INSERT |UPDATE |DELETE |CREATE |DROP |ALTER )", re.IGNORECASE),
        ),
        ("bash", re.compile(r"^(#!/|if \[|for |while |echo |export |source )")),
        ("java", re.compile(r"^(public |private |protected |class |interface |import java)")),
        ("c", re.compile(r"^(#include|int main|void |typedef )")),
    ]
    _SQL_CONT = re.compile(
        r"^(FROM|WHERE|AND|OR|JOIN|ORDER BY|GROUP BY|LIMIT|VALUES|SET)\b", re.IGNORECASE
    )

    def _detect_lang(lines: list) -> str:
        for line in lines[:3]:
            stripped = line.lstrip()
            for lang, pat in _LANG_SIGNALS:
                if pat.match(stripped):
                    return lang
        return ""

    def _is_code_start(line: str) -> bool:
        # 4-space-indented lines with code-like content are code starts
        if line.startswith("    ") or line.startswith("\t"):
            stripped = line.strip()
            if re.search(r"[=()\[\]{}]", stripped):
                return True
        stripped = line.lstrip()
        return any(pat.match(stripped) for _, pat in _LANG_SIGNALS)

    def _is_code_continuation(line: str) -> bool:
        if not line.strip():
            return True
        if line.startswith("    ") or line.startswith("\t"):
            return True
        stripped = line.strip()
        if stripped.startswith(
            ("#", "//", "/*", "*", "return ", "print(", "println!", "assert", "console.")
        ):
            return True
        if re.search(r"[{}()\[\];]", stripped):
            return True
        if _SQL_CONT.match(stripped):
            return True
        return False

    lines = text.split("\n")
    result: list = []
    in_fence = False
    code_run: list = []

    def flush_run() -> None:
        while code_run and not code_run[-1].strip():
            code_run.pop()
        if len(code_run) < 2:
            result.extend(code_run)
            code_run.clear()
            return
        lang = _detect_lang(code_run)
        # Ensure a blank line before the opening fence for proper markdown rendering.
        if result and result[-1].strip():
            result.append("")
        result.append(f"```{lang}")
        result.extend(code_run)
        result.append("```")
        code_run.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            if code_run:
                flush_run()
            in_fence = not in_fence
            result.append(line)
            i += 1
            continue
        if in_fence:
            result.append(line)
            i += 1
            continue
        if _is_code_start(line):
            code_run.append(line)
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if next_line.lstrip().startswith("```"):
                    break
                if (
                    next_line.strip()
                    and not _is_code_continuation(next_line)
                    and not _is_code_start(next_line)
                ):
                    break
                code_run.append(next_line)
                i += 1
            flush_run()
        else:
            if code_run:
                flush_run()
            result.append(line)
            i += 1
    if code_run:
        flush_run()
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Bullet normalization
# ---------------------------------------------------------------------------

_BULLET_CHAR_RE = re.compile(r"^([ \t]*)•\s*", re.MULTILINE)


def _normalize_bullets(text: str) -> str:
    """Replace • bullet characters with markdown - list syntax."""
    return _BULLET_CHAR_RE.sub(lambda m: m.group(1) + "- ", text)


# ---------------------------------------------------------------------------
# Math ASCII → Unicode converter
# ---------------------------------------------------------------------------

_SUP_MAP: dict = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    "+": "⁺",
    "-": "⁻",
    "(": "⁽",
    ")": "⁾",
}
_SUB_MAP: dict = {str(i): c for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")}

_GREEK_WORDS: list = [
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("epsilon", "ε"),
    ("zeta", "ζ"),
    ("eta", "η"),
    ("theta", "θ"),
    ("iota", "ι"),
    ("kappa", "κ"),
    ("lambda", "λ"),
    ("mu", "μ"),
    ("nu", "ν"),
    ("xi", "ξ"),
    ("pi", "π"),
    ("rho", "ρ"),
    ("sigma", "σ"),
    ("tau", "τ"),
    ("upsilon", "υ"),
    ("phi", "φ"),
    ("chi", "χ"),
    ("psi", "ψ"),
    ("omega", "ω"),
    ("Alpha", "Α"),
    ("Beta", "Β"),
    ("Gamma", "Γ"),
    ("Delta", "Δ"),
    ("Epsilon", "Ε"),
    ("Zeta", "Ζ"),
    ("Eta", "Η"),
    ("Theta", "Θ"),
    ("Iota", "Ι"),
    ("Kappa", "Κ"),
    ("Lambda", "Λ"),
    ("Mu", "Μ"),
    ("Nu", "Ν"),
    ("Xi", "Ξ"),
    ("Pi", "Π"),
    ("Rho", "Ρ"),
    ("Sigma", "Σ"),
    ("Tau", "Τ"),
    ("Upsilon", "Υ"),
    ("Phi", "Φ"),
    ("Chi", "Χ"),
    ("Psi", "Ψ"),
    ("Omega", "Ω"),
]
_GREEK_PATTERNS = [(re.compile(r"\b" + word + r"\b"), sym) for word, sym in _GREEK_WORDS]


def _mathify(text: str) -> str:
    """Convert ASCII math notation to Unicode scientific symbols.
    Handles superscripts, subscripts, Greek words, multiplication dots,
    comparison operators, and infinity keyword.
    """

    def _to_sup(chars: str) -> str:
        out = ""
        for c in chars:
            if c in _SUP_MAP:
                out += _SUP_MAP[c]
            else:
                return ""
        return out

    def _brace_sup(m: re.Match) -> str:
        converted = _to_sup(m.group(2))
        if converted:
            return m.group(1) + converted
        return m.group(0)

    def _simple_sup(m: re.Match) -> str:
        converted = _to_sup(m.group(2))
        if converted:
            return m.group(1) + converted
        return m.group(0)

    # Greek words first so that pi^2 → π^2 → π² (not pi² with unreplaced pi)
    for pat, sym in _GREEK_PATTERNS:
        text = pat.sub(sym, text)
    text = re.sub(r"(\w+)\^\{([^}]+)\}", _brace_sup, text)
    text = re.sub(r"(\w+)\^(\d+|\w)", _simple_sup, text)
    # Multiplication dot BEFORE subscript: I_0 * cos still has digit [0-9] before *
    text = re.sub(r"(?<=[a-zA-Z0-9]) \* (?=[a-zA-Z0-9])", " · ", text)
    text = re.sub(r"(?<=[a-zA-Z0-9])\*(?=[a-zA-Z0-9])", "·", text)
    text = re.sub(r"(\w)_(\d)", lambda m: m.group(1) + _SUB_MAP.get(m.group(2), m.group(2)), text)
    text = text.replace(">=", "≥").replace("<=", "≤").replace("!=", "≠")
    text = re.sub(r"\binfinity\b", "∞", text)
    return text


# ---------------------------------------------------------------------------
# Math formula detection and fencing
# ---------------------------------------------------------------------------

_MATH_OP_RE = re.compile(
    r"[\^]"
    r"|(?<=[a-zA-Z0-9]) \* (?=[a-zA-Z0-9])"
    r"|(?<=[a-zA-Z0-9])\*(?=[a-zA-Z0-9])"
    r"|sqrt\(|sin\(|cos\(|tan\(|log\(|exp\("
)
_LIST_ITEM_START_RE = re.compile(r"^[ \t]*[-*•][ \t]")


def _is_formula_line(line: str) -> bool:
    """Return True if line looks like a standalone math formula."""
    stripped = line.strip()
    if "=" not in stripped:
        return False
    if _LIST_ITEM_START_RE.match(line):
        return False
    if not _MATH_OP_RE.search(stripped):
        return False
    words = re.findall(r"[a-zA-Z]{3,}", stripped)
    _ENGLISH = {
        "the",
        "and",
        "for",
        "are",
        "was",
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "can",
        "all",
        "not",
        "but",
        "out",
        "more",
        "result",
        "value",
        "where",
        "which",
        "also",
    }
    if words:
        eng_count = sum(1 for w in words if w.lower() in _ENGLISH)
        if eng_count / len(words) > 0.4:
            return False
    return True


def _fence_math_formulas(text: str) -> str:
    """Wrap standalone math formula lines in code blocks with Unicode conversion."""
    lines = text.split("\n")
    result: list = []
    in_fence = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            result.append(line)
            i += 1
            continue
        if in_fence:
            result.append(line)
            i += 1
            continue
        if _is_formula_line(line):
            formula_run = [_mathify(line)]
            i += 1
            while i < len(lines) and _is_formula_line(lines[i]):
                formula_run.append(_mathify(lines[i]))
                i += 1
            result.append("```")
            result.extend(formula_run)
            result.append("```")
        else:
            result.append(line)
            i += 1
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Inline math symbols for prose lines
# ---------------------------------------------------------------------------


def _inline_math_symbols(text: str) -> str:
    """Apply Greek letter and subscript conversions to prose text outside fences."""
    lines = text.split("\n")
    result = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue
        for pat, sym in _GREEK_PATTERNS:
            line = pat.sub(sym, line)
        line = re.sub(
            r"(\w)_(\d)", lambda m: m.group(1) + _SUB_MAP.get(m.group(2), m.group(2)), line
        )
        result.append(line)
    return "\n".join(result)


def _preprocess_markdown(text: str) -> str:
    """Apply terminal-friendly substitutions before passing text to Rich Markdown.
    * ``- [x]`` / ``- [ ]`` → ✅ / ☐  (task lists)
    * ``~~text~~``           → ``text``   (strikethrough — unsupported in Rich)
    * ``> [!NOTE]`` etc.     → ``> **ℹ NOTE:** …`` (GitHub callout → blockquote)
    """

    def _task_sub(m: re.Match) -> str:
        checked = m.group(2).strip()
        return m.group(1) + ("✅ " if checked.lower() == "x" else "☐ ")

    text = _TASK_LIST_RE.sub(_task_sub, text)
    text = _STRIKETHROUGH_RE.sub(r"\1", text)

    def _callout_sub(m: re.Match) -> str:
        ctype = m.group(1)
        icon, color = _CALLOUT_RICH[ctype]
        body_lines = m.group(2)
        body = "\n".join(
            line[2:] if line.startswith("> ") else line for line in body_lines.rstrip().splitlines()
        )
        # Emit as a bold-prefixed blockquote — Rich renders ▌ border.
        prefix = f"**{icon} {ctype}:**"
        if body.strip():
            return f"> {prefix} {body.strip()}\n"
        return f"> {prefix}\n"

    text = _CALLOUT_RE.sub(_callout_sub, text)
    text = _normalize_bullets(text)
    text = _fence_math_formulas(text)
    text = _inline_math_symbols(text)
    text = _fence_unfenced_code(text)
    return text


# ---------------------------------------------------------------------------
# LaTeX → Unicode converter for terminal display
# ---------------------------------------------------------------------------

_LATEX_SYMBOLS: list[tuple[str, str]] = [
    # Greek letters — lowercase
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\gamma", "γ"),
    (r"\delta", "δ"),
    (r"\epsilon", "ε"),
    (r"\varepsilon", "ε"),
    (r"\zeta", "ζ"),
    (r"\eta", "η"),
    (r"\theta", "θ"),
    (r"\vartheta", "ϑ"),
    (r"\iota", "ι"),
    (r"\kappa", "κ"),
    (r"\lambda", "λ"),
    (r"\mu", "μ"),
    (r"\nu", "ν"),
    (r"\xi", "ξ"),
    (r"\pi", "π"),
    (r"\rho", "ρ"),
    (r"\sigma", "σ"),
    (r"\tau", "τ"),
    (r"\upsilon", "υ"),
    (r"\phi", "φ"),
    (r"\varphi", "φ"),
    (r"\chi", "χ"),
    (r"\psi", "ψ"),
    (r"\omega", "ω"),
    # Greek letters — uppercase
    (r"\Gamma", "Γ"),
    (r"\Delta", "Δ"),
    (r"\Theta", "Θ"),
    (r"\Lambda", "Λ"),
    (r"\Xi", "Ξ"),
    (r"\Pi", "Π"),
    (r"\Sigma", "Σ"),
    (r"\Upsilon", "Υ"),
    (r"\Phi", "Φ"),
    (r"\Psi", "Ψ"),
    (r"\Omega", "Ω"),
    # Operators and relations
    (r"\approx", "≈"),
    (r"\neq", "≠"),
    (r"\leq", "≤"),
    (r"\geq", "≥"),
    (r"\ll", "≪"),
    (r"\gg", "≫"),
    (r"\pm", "±"),
    (r"\mp", "∓"),
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\div", "÷"),
    (r"\circ", "∘"),
    (r"\infty", "∞"),
    (r"\partial", "∂"),
    (r"\nabla", "∇"),
    (r"\forall", "∀"),
    (r"\exists", "∃"),
    (r"\in", "∈"),
    (r"\notin", "∉"),
    (r"\subset", "⊂"),
    (r"\subseteq", "⊆"),
    (r"\supset", "⊃"),
    (r"\cup", "∪"),
    (r"\cap", "∩"),
    (r"\emptyset", "∅"),
    (r"\to", "→"),
    (r"\rightarrow", "→"),
    (r"\leftarrow", "←"),
    (r"\Rightarrow", "⇒"),
    (r"\Leftarrow", "⇐"),
    (r"\Leftrightarrow", "⇔"),
    (r"\leftrightarrow", "↔"),
    (r"\uparrow", "↑"),
    (r"\downarrow", "↓"),
    (r"\perp", "⊥"),
    (r"\parallel", "∥"),
    (r"\sim", "∼"),
    (r"\simeq", "≃"),
    (r"\equiv", "≡"),
    (r"\propto", "∝"),
    (r"\therefore", "∴"),
    (r"\because", "∵"),
    # Blackboard bold / script
    (r"\mathbb{R}", "ℝ"),
    (r"\mathbb{Z}", "ℤ"),
    (r"\mathbb{N}", "ℕ"),
    (r"\mathbb{Q}", "ℚ"),
    (r"\mathbb{C}", "ℂ"),
    (r"\mathbb{E}", "𝔼"),
    (r"\mathcal{O}", "𝒪"),
    # Functions
    (r"\log", "log"),
    (r"\ln", "ln"),
    (r"\exp", "exp"),
    (r"\sin", "sin"),
    (r"\cos", "cos"),
    (r"\tan", "tan"),
    (r"\min", "min"),
    (r"\max", "max"),
    (r"\sup", "sup"),
    (r"\inf", "inf"),
    (r"\lim", "lim"),
    (r"\det", "det"),
    (r"\text{sign}", "sign"),
    (r"\text{argmin}", "argmin"),
    (r"\text{argmax}", "argmax"),
    # Sums, integrals, products
    (r"\sum", "∑"),
    (r"\prod", "∏"),
    (r"\int", "∫"),
    (r"\iint", "∬"),
    (r"\oint", "∮"),
    # Dots
    (r"\ldots", "…"),
    (r"\cdots", "⋯"),
    (r"\vdots", "⋮"),
    (r"\ddots", "⋱"),
    # Brackets / norms
    (r"\langle", "⟨"),
    (r"\rangle", "⟩"),
    (r"\|", "‖"),
    (r"\left\|", "‖"),
    (r"\right\|", "‖"),
    (r"\left|", "|"),
    (r"\right|", "|"),
    # Misc
    (r"\sqrt", "√"),
    (r"\prime", "′"),
    (r"^\prime", "′"),
    (r"\dag", "†"),
    (r"\top", "ᵀ"),
    (r"^\top", "ᵀ"),
]

_SUPERSCRIPT_MAP = str.maketrans(
    "0123456789+-=()naebijrstuvwxyz",
    "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵃᵉᵇⁱʲʳˢᵗᵘᵛʷˣʸᶻ",
)
_SUBSCRIPT_MAP = str.maketrans(
    "0123456789+-=()aeijnorstuvx",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₙₒᵣₛₜᵤᵥₓ",
)

_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_SUP_RE = re.compile(r"\^\{([^{}]{1,20})\}|\^([A-Za-z0-9])")
_SUB_RE = re.compile(r"_\{([^{}]{1,20})\}|_([A-Za-z0-9])")
_CMD_ARG_RE = re.compile(r"\\[a-zA-Z]+\{([^{}]*)\}")


def _latex_to_unicode(formula: str) -> str:
    """Convert a LaTeX formula string to a Unicode-rich terminal representation.
    Handles Greek letters, operators, blackboard bold, fractions, super/subscripts,
    and common functions.  Falls back to the raw token for unknown commands.
    """
    s = formula
    # 1. Substitute named symbols — longest key first to avoid partial matches
    #    (e.g. \infty before \in, \int before \in, \iint before \int).
    for latex, uni in sorted(_LATEX_SYMBOLS, key=lambda t: len(t[0]), reverse=True):
        s = s.replace(latex, uni)
    # 2. \\frac{num}{den} → num/den
    s = _FRAC_RE.sub(lambda m: f"{m.group(1)}/{m.group(2)}", s)

    # 3. Superscripts  ^{...} or ^x
    def _sup(m: re.Match) -> str:
        inner = m.group(1) or m.group(2)
        return inner.translate(_SUPERSCRIPT_MAP)

    s = _SUP_RE.sub(_sup, s)

    # 4. Subscripts  _{...} or _x
    def _sub(m: re.Match) -> str:
        inner = m.group(1) or m.group(2)
        return inner.translate(_SUBSCRIPT_MAP)

    s = _SUB_RE.sub(_sub, s)
    # 5. Strip remaining \cmd{arg} wrappers → keep the arg (e.g. \text{foo} → foo)
    s = _CMD_ARG_RE.sub(lambda m: m.group(1), s)
    # 6. Remove remaining backslash commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # 7. Convert \sqrt{...} → √(...)
    s = re.sub(r"√\{([^{}]*)\}", r"√(\1)", s)
    # 8. Strip remaining bare braces (artifacts from nested commands)
    s = s.replace("{", "").replace("}", "")
    # 9. Strip leftover bare backslashes and tidy whitespace
    s = s.replace("\\", "").strip()
    s = re.sub(r" {2,}", " ", s)
    return s


def _make_math_renderable(text: str):  # type: ignore[return]
    """Return a Rich renderable for *text* with math formula support.
    Block formulas (``$$...$$``) become cyan-bordered panels with LaTeX
    symbols converted to Unicode (Greek letters, operators, fractions, etc.).
    Inline formulas (``$...$``) are converted to Unicode and wrapped in
    backticks so Markdown renders them as code spans.
    Returns a single Rich renderable suitable for ``Live.update()``
    or ``Console.print()``.
    """
    from rich.console import Group as _RGroup
    from rich.markdown import Markdown as _RM
    from rich.panel import Panel as _RP
    from rich.text import Text as _RT

    # Apply task-list / strikethrough / callout substitutions first.
    text = _preprocess_markdown(text)
    # Split on $$...$$ — result alternates [text, formula, text, formula, ...]
    segments = _BLOCK_MATH_RE.split(text)
    renderables = []
    for i, seg in enumerate(segments):
        if i % 2 == 1:
            # Block math: convert to Unicode and display in a bordered panel.
            formula = _latex_to_unicode(seg.strip())
            renderables.append(
                _RP(
                    _RT(formula, style="bold cyan"),
                    title="[dim cyan]∫ math[/dim cyan]",
                    border_style="dim cyan",
                    padding=(0, 1),
                )
            )
        elif seg.strip():
            # Regular text: convert inline $...$ to Unicode, then wrap in backticks
            # so Rich Markdown displays them as code spans with visual distinction.
            def _replace_inline(m: re.Match) -> str:
                return "`" + _latex_to_unicode(m.group(1)) + "`"

            styled = _INLINE_MATH_RE.sub(_replace_inline, seg)
            renderables.append(_RM(styled, code_theme=_MD_CODE_THEME))
    if not renderables:
        return _RM(text, code_theme=_MD_CODE_THEME)
    if len(renderables) == 1:
        return renderables[0]
    return _RGroup(*renderables)


def _render_rich_with_math(text: str, console) -> None:  # type: ignore[type-arg]
    """Print *text* with math support via :func:`_make_math_renderable`."""
    console.print(_make_math_renderable(text))


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
        print("    ⚠️  No GitHub OAuth token set for the 'github_copilot' provider.")
        print("    Starting GitHub OAuth device flow…")
        try:
            key = _copilot_device_flow()
        except (EOFError, KeyboardInterrupt, RuntimeError) as e:
            print(f"\n    Cancelled: {e}")
            return False
        _save_env_key(env_name, key)
        brain.config.copilot_pat = key
        # Clear cached session so the new OAuth token is exchanged on next use
        brain.llm._openai_clients.pop("_copilot", None)
        brain.llm._openai_clients.pop("_copilot_session", None)
        brain.llm._openai_clients.pop("_copilot_token", None)
        print(f"    {env_name} saved and applied.")
        return True
    print(f"    ⚠️  No {env_name} set for the '{provider}' provider.")
    try:
        import getpass

        key = getpass.getpass(f"  Enter {env_name} (hidden, Enter to skip): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n    Skipped.")
        return False
    if not key:
        print("    Skipped — you can set it later with /keys set " + provider)
        return False
    _save_env_key(env_name, key)
    setattr(brain.config, attr, key)
    print(f"    {env_name} saved and applied.")
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
    TOP = f"    ╭{'─' * W}╮"
    BOTTOM = f"    ╰{'─' * W}╯"
    SEP = f"    ├{'─' * W}┤"
    BLANK = f"    │{' ' * W}│"

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
        print("    Nothing to compact — chat history is empty.")
        return
    turns_before = len(chat_history)
    print(f"    ⠿ Compacting {turns_before} turns…", end="", flush=True)
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
        print(
            f"\r    Compacted {turns_before} turns -> 1 summary  (~{tokens_saved:,} tokens freed)"
        )
    except Exception as e:
        print(f"\r    Compact failed: {e}")


# ── Banner constants ───────────────────────────────────────────────────────────

_HINT = "    Type your question  ·  /help for commands  ·  Tab to autocomplete  ·  @file or @folder/ to attach context"


def _box_width() -> int:
    """Return inner box width: terminal columns minus 4, minimum 43."""
    return max(43, shutil.get_terminal_size((120, 24)).columns - 6)


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
    return f"    ┃{content}{' ' * pad}┃"


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
    blank = f"    ┃{' ' * bw}┃"
    rows = [
        f"    \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m",  # 1
        blank,  # 2
        *[  # 3-8  blue-shaded art lines + brain design
            f"    ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{_get_brain_anim_row(i, -1, apad_w)}┃"
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
    rows.append(f"    \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m")  # 13
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
    sep = "    " + "─" * bw
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
            sys.stdout.write(f"    \033[1;32mYou\033[0m: {content}\n")
        elif role == "assistant":
            # Cap very long responses so they don't flood the screen
            if len(content) > 600:
                content = content[:600] + "…"
            sys.stdout.write(f"\n    \033[1;33mAxon\033[0m:\n    {content}\n")
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
        # Redirect stderr to /dev/null for the duration of the animation so
        # that library noise (e.g. transformers "UNEXPECTED key" warnings) does
        # not write bytes to the terminal and corrupt the alternate screen.
        self._real_stderr = sys.stderr
        try:
            sys.stderr = open(os.devnull, "w", encoding="utf-8")
        except Exception:
            pass
        bw = _box_width()
        art_pad = " " * max(0, bw - 39)  # 4 indent + 35 art cols = 39 vis cols
        _art_rows = "".join(
            f"\r    ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{art_pad}┃\n"
            for i, line in enumerate(_AXON_ART)
        )
        # Enter the alternate screen buffer: gives us a completely clean canvas
        # so we can redraw from \x1b[H (top-left) every frame with zero cursor-
        # tracking arithmetic.  Primary screen content is preserved and restored
        # automatically when we exit with \x1b[?1049l in stop().
        sys.stdout.write(
            "\x1b[?1049h"  # enter alternate screen
            "\x1b[?25l"  # hide cursor
            "\x1b[H"  # cursor to top-left (1, 1)
            f"\r\n"  # blank line for top margin
            f"\r    \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m\n"
            f"\r    ┃{' ' * bw}┃\n" + _art_rows + f"\r    ┃{' ' * bw}┃\n"
            f"\r    ┃{'    ⠿  Initializing…'.ljust(bw)}┃\n"
            f"\r    ┃{' ' * bw}┃\n"
            f"\r    \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m\n"
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
                art_rows = "".join(
                    f"\r    ┃    {_AXON_BLUE[i]}{_AXON_ART[i]}{_AXON_RST}"
                    f"{_anim_pad(i, self._anim_frame, apad_w)}┃\n"
                    for i in range(6)
                )
                if self._step:
                    spinner = self._FRAMES[self._idx % len(self._FRAMES)]
                    step_text = f"    {spinner}  {self._step}"
                    self._idx += 1
                elif self.tick_lines:
                    step_text = f"    ✓  {self.tick_lines[-1]}"
                else:
                    step_text = "    ⠿  Initializing…"
                step_line = f"\r    ┃{step_text.ljust(bw)}┃\n"
                # Cursor to home on the alternate screen, then redraw the full
                # box from scratch.  No cursor arithmetic needed at all — we
                # always know exactly where we are.
                buf = (
                    "\x1b[H"  # cursor to (1,1) on alternate screen
                    "\r\n"  # blank top-margin line
                    f"\r    \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m\n"
                    f"\r    ┃{' ' * bw}┃\n"
                    + art_rows
                    + f"\r    ┃{' ' * bw}┃\n"
                    + step_line
                    + f"\r    ┃{' ' * bw}┃\n"
                    f"\r    \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m\n"
                )
                sys.stdout.write(buf)
                sys.stdout.flush()

    def _tick(self, label: str) -> None:
        with self._lock:
            self._step = ""
            self.tick_lines.append(label)
            # The next spin_loop frame redraws the whole box showing "✓ label"
            # in the step line — no relative cursor positioning needed here.

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
        # Restore stderr before any further output
        try:
            sys.stderr.close()
        except Exception:
            pass
        sys.stderr = self._real_stderr
        # Exit alternate screen — the primary screen with its original content
        # is restored automatically by the terminal.  Also restore the cursor.
        sys.stdout.write("\x1b[?1049l\x1b[?25h")
        sys.stdout.flush()


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
        raw_path = m.group(1).rstrip("/\\")
        from pathlib import Path as _P

        p = _P(raw_path)
        try:
            p_resolved = p.resolve()
        except Exception:
            p_resolved = p
        if p_resolved.is_dir():
            path_str = str(p_resolved)
            return f"\n\n=== folder: {path_str} ===\n{_expand_dir(path_str)}\n=== end folder ===\n"
        if p_resolved.is_file():
            path_str = str(p_resolved)
            content = _read_file(path_str)
            if content:
                return f"\n\n--- @{path_str} ---\n{content}\n--- end ---\n"
        return m.group(0)

    return re.sub(r"(?<!\S)@(\S+)", _replace, text)


# ---------------------------------------------------------------------------

# /config command handler

# ---------------------------------------------------------------------------


def _handle_config_cmd(arg: str, brain, cfg_path: str) -> None:
    """Handle /config sub-commands.
    Sub-commands:
      (empty) / show   — render config table
      validate          — run AxonConfig.validate() and show issues
      wizard            — launch interactive setup wizard
      reset             — overwrite config.yaml with defaults
      set <key> <value> — set a dot-notation field
    """
    from axon.config import AxonConfig
    from axon.config_wizard import render_config_table, render_issues, run_wizard

    parts = arg.split(None, 2)
    subcmd = parts[0].lower() if parts else ""
    if subcmd in ("", "show"):
        if brain is None:
            print("    Brain not initialised.")
            return
        render_config_table(brain.config)
    elif subcmd == "validate":
        issues = AxonConfig.validate(cfg_path or None)
        render_issues(issues)
    elif subcmd == "wizard":
        if brain is None:
            print("    Brain not initialised.")
            return
        try:
            changes = run_wizard(brain=brain, config_path=cfg_path)
        except KeyboardInterrupt:
            print("\n    Setup cancelled.")
            return
        if changes:
            for key, val in changes.items():
                setattr(brain.config, key, val)
            brain.config.save(cfg_path or None)
            print(f"    Saved {len(changes)} change(s) to {cfg_path or '(default path)'}.")
    elif subcmd == "reset":
        try:
            confirm = input("  Overwrite config.yaml with defaults? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"
        if confirm == "y":
            # use module-level os
            from pathlib import Path

            from axon.config import _DEFAULT_CONFIG_YAML, _USER_CONFIG_PATH

            target = cfg_path or str(_USER_CONFIG_PATH)
            os.makedirs(os.path.dirname(os.path.expanduser(target)), exist_ok=True)
            Path(os.path.expanduser(target)).write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
            print(f"    Config reset to defaults at {target}")
        else:
            print("    Cancelled.")
    elif subcmd == "set":
        if len(parts) < 3:
            print("    Usage: /config set <key> <value>")
            print("    Example: /config set chunk.strategy markdown")
            return
        from axon.api_routes.config_routes import _DOT_TO_FLAT

        dot_key = parts[1]
        raw_val: str = parts[2]
        flat_key = _DOT_TO_FLAT.get(dot_key, dot_key.replace(".", "_"))
        if brain is None:
            print("    Brain not initialised.")
            return
        if not hasattr(brain.config, flat_key):
            print(f"    Unknown config key '{dot_key}'. Known keys: {sorted(_DOT_TO_FLAT.keys())}")
            return
        # Coerce value type based on existing attribute type
        current = getattr(brain.config, flat_key)
        try:
            if isinstance(current, bool):
                coerced = raw_val.lower() in ("true", "1", "yes")
            elif isinstance(current, int):
                coerced = int(raw_val)
            elif isinstance(current, float):
                coerced = float(raw_val)
            else:
                coerced = raw_val
        except (ValueError, TypeError):
            coerced = raw_val
        setattr(brain.config, flat_key, coerced)
        brain.config.save(cfg_path or None)
        print(f"    Set {dot_key} = {coerced!r}  (saved to config).")
    else:
        print(
            f"    Unknown sub-command '{subcmd}'. "
            "Usage: /config [show|validate|wizard|reset|set <key> <value>]"
        )


def _find_bash() -> list[str] | None:
    """Detect the best available bash interpreter on the current platform.
    Returns a command prefix suitable for ``subprocess.run([*prefix, cmd])``,
    or ``None`` if no bash is found (caller should fall back to ``shell=True``).
    Detection order:
    1. ``bash`` in PATH  — covers Linux, macOS, and Windows with Git Bash in PATH.
    2. ``wsl`` in PATH   — Windows Subsystem for Linux (Windows only).
    3. Known Git Bash install paths (Windows only).
    """
    import shutil as _shutil

    if _shutil.which("bash"):
        return ["bash", "-c"]
    if sys.platform == "win32":
        if _shutil.which("wsl"):
            return ["wsl", "bash", "-c"]
        for _p in [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]:
            if os.path.exists(_p):
                return [_p, "-c"]
    return None


def _resolve_bash(setting: str) -> list[str] | None:
    """Return bash command prefix according to ``repl.shell`` config value."""
    import shutil as _shutil

    if setting == "native":
        return None
    if setting == "wsl":
        return ["wsl", "bash", "-c"] if _shutil.which("wsl") else None
    if setting == "gitbash":
        for _p in [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ]:
            if os.path.exists(_p):
                return [_p, "-c"]
        return None
    if setting == "bash":
        b = _shutil.which("bash")
        return [b, "-c"] if b else None
    # "auto" (default)
    return _find_bash()


def _interactive_repl(
    brain: AxonBrain,
    stream: bool = True,
    init_display: _InitDisplay | None = None,
    quiet: bool = False,
    _scripted_inputs: list[str] | None = None,
) -> None:
    """Interactive REPL chat session with session persistence and live tab completion.
    Features:
    - Session persistence: auto-saves to ~/.axon/sessions/session_<timestamp>.json
    - Live tab completion: slash commands, filesystem paths, Ollama model names via prompt_toolkit
    - Animated spinners: braille spinner during init and LLM generation (disabled in quiet mode)
    - Slash commands: /help, /list, /ingest, /model, /embed, /pull, /search, /discuss, /rag,
      /compact, /context, /sessions, /resume, /retry, /clear, /project, /agent, /keys, /vllm-url, /quit, /exit
    - @file/folder context: type @path/file.txt or @path/folder/ to inline contents into your query (read-only)
    - Shell passthrough: !command runs a shell command (local-only by default)
    - Pinned status info: token usage, model info, RAG settings visible at terminal bottom
    Args:
        brain: AxonBrain instance to use for queries.
        stream: If True, streams LLM response token-by-token; if False, waits for full response.
        init_display: Optional _InitDisplay handler to stop after initialization.
        quiet: Suppress spinners and progress bars (auto-enabled for non-TTY stdin).
    """
    from axon.logging_setup import configure_logging

    configure_logging()

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
        "google_genai",
        "google.genai",
    ):
        _lg = _logging.getLogger(_log)
        _lg.setLevel(_logging.WARNING)
        _lg.propagate = False  # prevent bubbling to root logger
    # ── Input: prefer prompt_toolkit (live completions), fall back to readline ──
    _pt_app = None
    _input_queue = None
    _repl_event_loop = None  # captured inside _repl_loop_async for thread-safe run_in_terminal
    _spin_state: dict = {"active": False, "idx": 0}
    try:
        import glob as _pglob

        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import has_completions
        from prompt_toolkit.formatted_text import ANSI as _PTANSI
        from prompt_toolkit.formatted_text import HTML as _PThtml
        from prompt_toolkit.formatted_text import FormattedText as _PTFT  # noqa: F401
        from prompt_toolkit.history import FileHistory as _FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.key_binding.bindings.emacs import load_emacs_bindings
        from prompt_toolkit.layout import Layout
        from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.patch_stdout import patch_stdout as _pt_patch_stdout
        from prompt_toolkit.styles import Style

        _HIST_DIR = os.path.expanduser("~/.axon")
        os.makedirs(_HIST_DIR, exist_ok=True)
        _HIST_FILE = os.path.join(_HIST_DIR, "repl_history")
        _PT_STYLE = Style.from_dict(
            {
                "": "",
                # Completion menu — dark panel with highlighted selection
                "completion-menu": "bg:#1e2030 #cdd6f4",
                "completion-menu.completion": "bg:#1e2030 #cdd6f4",
                "completion-menu.completion.current": "bg:#363a4f #cba6f7 bold",
                "completion-menu.meta.completion": "bg:#1e2030 #6c7086 italic",
                "completion-menu.meta.completion.current": "bg:#363a4f #89b4fa italic",
                "completion-menu.border": "#494d64",
                # Input area
                "bottom-toolbar": "noinherit bg:default fg:#87ceeb",
                "separator": "#334466",
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
                            desc = _SLASH_CMD_DESC.get(c, "")
                            yield Completion(c[len(text) :], display=c, display_meta=desc)
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
                        # use module-level os
                        if os.getenv("AXON_DRY_RUN"):
                            pass
                        else:
                            try:
                                import ollama as _ol

                                resp = _ol.list()
                                mods = (
                                    resp.models
                                    if hasattr(resp, "models")
                                    else resp.get("models", [])
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
                    sub_opts = ["list", "new ", "switch ", "delete ", "folder", "refresh"]
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
            # Fast-path: if embedding is in progress, show the progress bar immediately
            # without hitting ChromaDB (which may be locked during ingest).
            from axon._ui_state import state as _ui_state_fast

            _embed_prog_fast = _ui_state_fast.get("embed_progress", "")
            if _embed_prog_fast:
                _BON = "\x1b[1m"
                _RST = "\x1b[0m"
                m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
                emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
                row1 = f"    {_BON}LLM\x1b[22m  {m}    {_BON}Embed\x1b[22m  {emb}"
                return _PTANSI(f"{row1}\n    \x1b[1;32m{_embed_prog_fast}\x1b[0m{_RST}")
            if _spin_state.get("active"):
                _tb_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                _tf = _tb_frames[_spin_state.get("idx", 0) % len(_tb_frames)]
                return _PTANSI(
                    f"    \x1b[1;33m✦\x1b[0m \x1b[2m{_tf} Thinking\u2026  Ctrl+C to cancel\x1b[0m"
                )

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
                f"    {_lbl('LLM')}  {_t(m, W1):{W1}}{sep}"
                f"{_lbl('Embed')}  {_t(emb, W2):{W2}}{sep}"
                f"{_lbl('Docs')}  {doc_s}"
            )
            row2 = (
                f"    {_lbl('search')}:{s_state}{_pad('search', s_state, C1)}{sep}"
                f"{_lbl('discuss')}:{d_state}{_pad('discuss', d_state, C2)}{sep}"
                f"{_lbl('hybrid')}:{h_state}  "
                f"{_lbl('top-k')}:{brain.config.top_k}  "
                f"{_lbl('thr')}:{brain.config.similarity_threshold}"
                f"{proj_s}"
            )
            return _PTANSI(f"{row1}\n{row2}{_RST}")

        # Build the Application when running interactively on a real TTY.
        # Skip in test-mode (_scripted_inputs). Attempt to create a full-screen
        # prompt_toolkit Application and fall back gracefully to readline/input
        # if the environment does not support it.
        if _scripted_inputs is None:
            try:

                def _handle_enter(buf: Buffer) -> None:
                    text = buf.text
                    if text.strip() and _input_queue is not None:
                        _input_queue.put_nowait(text)
                    # Do NOT call buf.reset() here — prompt_toolkit resets the buffer
                    # after accept_handler returns and records history before reset.

                _input_buf = Buffer(
                    completer=_PTCompleter(brain),
                    complete_while_typing=True,
                    history=_FileHistory(_HIST_FILE),
                    name="input",
                    accept_handler=_handle_enter,
                )
                _kb = KeyBindings()

                @_kb.add("enter")
                def _handle_enter_key(event):
                    # validate_and_handle() calls accept_handler then resets the buffer.
                    # This is what PromptSession does internally; we must wire it
                    # explicitly in a raw Application since load_emacs_bindings()
                    # does not include the submit-on-enter behaviour.
                    event.current_buffer.validate_and_handle()

                @_kb.add("c-c")
                def _handle_ctrl_c(event):
                    if _input_buf.text:
                        _input_buf.reset()
                    else:
                        event.app.exit(exception=KeyboardInterrupt())

                @_kb.add("c-d")
                def _handle_ctrl_d(event):
                    event.app.exit(exception=EOFError())

                from prompt_toolkit.key_binding import merge_key_bindings

                _SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                _pt_app = Application(
                    layout=Layout(
                        HSplit(
                            [
                                # Completions expand upward above the input row
                                ConditionalContainer(
                                    content=CompletionsMenu(max_height=8, scroll_offset=1),
                                    filter=has_completions,
                                ),
                                # Thin separator line between conversation and input area
                                Window(
                                    height=1,
                                    content=FormattedTextControl(
                                        lambda: "    "
                                        + "─" * max(1, shutil.get_terminal_size().columns - 8)
                                    ),
                                    style="class:separator",
                                ),
                                Window(
                                    content=BufferControl(
                                        buffer=_input_buf,
                                        focusable=True,
                                        include_default_input_processors=True,
                                    ),
                                    get_line_prefix=lambda lineno, wrap_count: _PThtml(
                                        "    <ansigreen><b>></b></ansigreen> "
                                    ),
                                    height=1,
                                    wrap_lines=False,
                                ),
                                # Thin separator between input and status bar
                                Window(
                                    height=1,
                                    content=FormattedTextControl(
                                        lambda: "    "
                                        + "─" * max(1, shutil.get_terminal_size().columns - 8)
                                    ),
                                    style="class:separator",
                                ),
                                Window(
                                    content=FormattedTextControl(_toolbar),
                                    height=2,
                                    style="class:bottom-toolbar",
                                ),
                            ]
                        )
                    ),
                    key_bindings=merge_key_bindings([load_emacs_bindings(), _kb]),
                    style=_PT_STYLE,
                    mouse_support=False,
                    full_screen=False,
                )
                # Store app reference so background threads can trigger redraws.
                from axon._ui_state import state as _ui_state_ref

                _ui_state_ref["_pt_app"] = _pt_app
            except Exception as _pt_exc:  # pragma: no cover - best-effort fallback
                import logging as _logging

                _logging.getLogger("Axon").info(
                    "Prompt toolkit Application unavailable: %s. Falling back to readline/input.",
                    _pt_exc,
                )
                _pt_app = None
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
            f"    \033[1mLLM\033[0m\033[2m  {_t(m, W1):{W1}}"
            f"{sep}\033[0m\033[1mEmbed\033[0m\033[2m  {_t(emb, W2):{W2}}"
            f"{sep}\033[0m\033[1mDocs\033[0m\033[2m  {doc_s}\033[0m"
        )
        row2 = (
            f"\033[2m    {s_val:<{C1}}" f"{sep}{d_val:<{C2}}" f"{sep}{h_val}  {tk}{proj_s}\033[0m"
        )
        print(row1)
        print(row2)

    # Shared iterator for test-mode scripted inputs (covers both main loop and
    # mid-command confirmation prompts like /clear and /project new).
    _script_iter = iter(_scripted_inputs) if _scripted_inputs is not None else None

    def _read_input(prompt: str = "") -> str:
        # During Application.run_in_terminal the terminal is given back to us,
        # so plain input() works correctly here.  In test-mode, consume from the
        # scripted iterator so mid-command prompts get the right canned answer.
        if _script_iter is not None:
            try:
                return next(_script_iter)
            except StopIteration:
                raise EOFError
        # If tests monkeypatch prompt_toolkit.PromptSession to provide scripted
        # inputs, prefer that API so pytest can capture stdout while supplying
        # input via the fake PromptSession. This preserves existing test
        # fixtures that replaced PromptSession.
        try:
            import prompt_toolkit as _pt

            PS = getattr(_pt, "PromptSession", None)
            if PS:
                try:
                    session = PS()
                    return session.prompt(prompt if prompt else "\033[1;32mYou\033[0m: ")
                except StopIteration:
                    raise EOFError
                except Exception:
                    # Fall back to builtin input() if PromptSession fails for any reason
                    pass
        except Exception:
            pass
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
    _agent_mode: bool = False
    # Initial snapshot to avoid printing status on the very first query
    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
    tk = f"top-k:{brain.config.top_k}"
    thr = f"thr:{brain.config.similarity_threshold}"
    _last_config_snapshot: tuple = (m, s_v, d_v, h_v, tk, thr)
    # Resolve bash once at REPL startup — used by both ! passthrough and run_shell agent tool.
    _bash_cmd: list[str] | None = _resolve_bash(
        str(getattr(brain.config, "repl_shell", "auto")).lower().strip()
    )

    def _process_input_sync(user_input: str) -> bool:
        """Process one REPL input. Returns True to exit the REPL loop."""
        nonlocal session, _agent_mode, _last_sources, _last_query, _last_config_snapshot
        # --- Shell passthrough: !command ---
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                allowed, policy = _shell_passthrough_allowed()
                if not allowed:
                    print(
                        "    Shell passthrough blocked by policy "
                        f"(repl.shell_passthrough={policy})."
                    )
                    return False
                import subprocess

                # Intercept `cd` — child subprocess cwd changes don't propagate to the
                # parent REPL process, so we call os.chdir() directly instead.
                _sc = shell_cmd.strip()
                if _sc == "cd" or _sc.startswith("cd "):
                    _target = _sc[2:].strip() or str(Path.home())
                    try:
                        os.chdir(Path(_target).expanduser())
                        print(f"    {os.getcwd()}")
                    except OSError as _cde:
                        print(f"    cd: {_cde}")
                    return False

                # Route through bash when available; fall back to cmd.exe / /bin/sh.
                # Use run_in_terminal so the Application suspends while the subprocess
                # writes to the raw TTY — prevents display corruption.
                def _run_shell_cmd() -> None:
                    if _bash_cmd:
                        subprocess.run([*_bash_cmd, shell_cmd], cwd=os.getcwd())
                    else:
                        subprocess.run(shell_cmd, shell=True)

                if _pt_app is not None and _repl_event_loop is not None:
                    import asyncio as _aio

                    from prompt_toolkit.application import in_terminal as _in_terminal

                    async def _run_shell_in_terminal() -> None:
                        async with _in_terminal():
                            _run_shell_cmd()

                    _fut = _aio.run_coroutine_threadsafe(
                        _run_shell_in_terminal(),
                        _repl_event_loop,
                    )
                    _fut.result()
                else:
                    _run_shell_cmd()
            return False
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
                return True
            elif cmd == "/help":
                if arg:
                    # Per-command detail
                    _detail = {
                        "model": "    /model <model>              keep current provider\n"
                        "    /model <provider>/<model>   switch provider + model\n"
                        "    providers: ollama, gemini, openai, ollama_cloud, vllm, github_copilot\n"
                        "    e.g.  /model gemini/gemini-2.0-flash\n"
                        "          /model ollama/gemma:2b\n"
                        "          /model openai/gpt-4o\n"
                        "          /model vllm/meta-llama/Llama-3.1-8B-Instruct",
                        "embed": "    /embed <model>              keep current provider\n"
                        "    /embed <provider>/<model>   switch provider + model\n"
                        "    /embed /path/to/local        local HuggingFace folder\n"
                        "    providers: sentence_transformers, ollama, fastembed, openai\n"
                        "    !  Re-ingest after changing embedding model.",
                        "ingest": "    /ingest <path>              ingest a directory\n"
                        "    /ingest ./src/*.py           glob pattern\n"
                        "    /ingest ./notes/**/*.md      recursive glob",
                        "llm": "    /llm                         show LLM settings (provider, model, temperature)\n"
                        "    /llm temperature <0.0–2.0>   set generation temperature\n"
                        "    Lower temperature = more deterministic; higher = more creative.",
                        "rag": "    /rag                         show all RAG settings\n"
                        "    /rag topk <n>                results to retrieve (1–20)\n"
                        "    /rag threshold <0.0–1.0>     min similarity score\n"
                        "    /rag hybrid                  toggle hybrid BM25+vector\n"
                        "    /rag rerank                  toggle cross-encoder reranker\n"
                        "    /rag rerank-model <model>    set reranker model (HF ID or local path)\n"
                        "    /rag hyde                    toggle HyDE query expansion\n"
                        "    /rag multi                   toggle multi-query expansion\n"
                        "    /rag step-back               toggle step-back prompting\n"
                        "    /rag decompose               toggle query decomposition\n"
                        "    /rag compress                toggle LLM context compression\n"
                        "    /rag cite                    toggle inline [Document N] citations\n"
                        "    /rag raptor                  toggle RAPTOR hierarchical indexing\n"
                        "    /rag graph-rag               toggle GraphRAG entity retrieval",
                        "sessions": "    /sessions                    list recent saved sessions\n"
                        "    /resume <id>                 load a session by ID\n"
                        "    Sessions auto-save after each turn.",
                        "keys": "    /keys                        show API key status for all providers\n"
                        "    /keys set <provider>         interactively set an API key\n"
                        "    providers: gemini, openai, brave, ollama_cloud\n"
                        "    Keys are saved to ~/.axon/.env and loaded at startup.",
                        "project": "    /project                               show active project + list all\n"
                        "    /project list                          list all projects and mounted shares\n"
                        "    /project new <name>                    create a new project and switch to it\n"
                        "    /project new <name> <desc>             create with description\n"
                        "    /project new <parent>/<child>          create a sub-project (up to 5 levels)\n"
                        "    /project switch <name>                 switch to an existing local project\n"
                        "    /project switch <parent>/<child>       switch to a sub-project\n"
                        "    /project switch default                return to the global knowledge base\n"
                        "    /project switch @projects              merged view of all local projects\n"
                        "    /project switch @mounts                merged view of all mounted shares\n"
                        "    /project switch @store                 merged view of entire AxonStore\n"
                        "    /project switch mounts/<name>          switch to a mounted share\n"
                        "    /project delete <name>                 delete a leaf project and its data\n"
                        "    /project folder                        open the active project folder\n"
                        "    /project rotate-keys [<name>]          rotate sealed project DEK; invalidates all shares\n"
                        "\n"
                        "    Projects are stored in ~/.axon/projects/<name>/\n"
                        "    Sub-projects use nested subs/ directories (max depth: 5).\n"
                        "    Switching to a parent project shows merged data across all sub-projects.\n"
                        "    Use /ingest after switching to add documents to the current project.",
                        "share": "    /share list                              list all issued and received shares\n"
                        "    /share generate <project> <grantee>             generate a read-only share key\n"
                        "                                                    (auto-detects sealed projects)\n"
                        "    /share redeem <share_string>                    mount a shared project\n"
                        "                                                    (auto-detects sealed envelopes)\n"
                        "    /share revoke <key_id>                                  revoke a legacy share\n"
                        "    /share revoke <ssk_id> --project <name>                 sealed soft revoke\n"
                        "    /share revoke <ssk_id> --project <name> --rotate        sealed hard revoke (rotates DEK)\n"
                        "\n"
                        "    Share strings are base64-encoded payloads; send them out-of-band.\n"
                        "    Mounted shares appear under mounts/ in your project list and can be\n"
                        "    switched to with /project switch mounts/<name>.",
                        "store": "    /store whoami                            show store identity and active project\n"
                        "    /store init <base_path>                  change the store base path\n"
                        "    /store status                            show sealed-store init/unlock state\n"
                        "    /store bootstrap <passphrase>            one-time sealed-store init\n"
                        "    /store unlock <passphrase>               unlock for sealed-project queries\n"
                        "    /store lock                              clear in-process master cache\n"
                        "    /store change-passphrase <old> <new>     rotate the sealed-store passphrase\n"
                        "\n"
                        "    Example: /store init ~/axon_data\n"
                        "    Moves data to: <base_path>/AxonStore/<username>/\n"
                        "    Config is updated and persisted to ~/.config/axon/config.yaml.\n"
                        "    Sealed-store sub-commands require the [sealed] extra installed.",
                        "graph": "    /graph status                  show entity count, edges, community summaries\n"
                        "    /graph finalize                trigger community detection rebuild\n"
                        "    /graph conflicts               list conflicted facts (dynamic_graph backend)\n"
                        "    /graph retrieve <q> [--at TS]  run backend.retrieve directly (point-in-time)\n"
                        "    /graph viz [path]              export graph as HTML (opens in browser)\n"
                        "\n"
                        "    GraphRAG must be enabled: /rag graph-rag\n"
                        "    Finalize is useful after batch ingest with community summarisation deferred.",
                        "refresh": "    /refresh                       re-ingest files whose content has changed\n"
                        "\n"
                        "    Computes current content hash for each tracked file and compares\n"
                        "    to the stored hash from the last ingest. Only changed files are re-ingested.\n"
                        "    Use /stale to preview which files are old before refreshing.",
                        "stale": "    /stale                         list documents older than 7 days\n"
                        "    /stale <days>                  list documents older than N days\n"
                        "\n"
                        "    Reports age based on the last ingest timestamp for each source.\n"
                        "    Use /refresh to re-ingest any changed files.",
                        "config": "    /config                        show current config as a table\n"
                        "    /config show                   same as /config\n"
                        "    /config validate               validate config.yaml and list issues\n"
                        "    /config wizard                 interactive setup wizard\n"
                        "    /config reset                  overwrite config.yaml with defaults\n"
                        "    /config set <key> <value>      set a dot-notation config key\n"
                        "\n"
                        "    Example: /config set chunk.strategy markdown\n"
                        "             /config set rag.top_k 15\n"
                        "             /config set llm.model gemma3:4b",
                        "theme": "    /theme                         show current markdown code-block theme\n"
                        "    /theme list                    list popular theme names\n"
                        "    /theme <name>                  switch to a Pygments theme by name\n"
                        "\n"
                        "    Choice is saved to ~/.axon/prefs.json and restored on next launch.\n"
                        "    Any valid Pygments style name is accepted (e.g. dracula, nord, vs).\n"
                        "    Default: monokai",
                    }
                    key = arg.lstrip("/")
                    if key in _detail:
                        print(f"\n{_detail[key]}\n")
                    else:
                        print(f"    No detail for '{arg}'. Available: {', '.join(_detail)}")
                else:
                    print(
                        "\n"
                        "    /agent          toggle agent mode (LLM can call Axon tools directly)\n"
                        "    /clear          clear knowledge base for current project\n"
                        "    /compact        summarise conversation to free context\n"
                        "    /config [sub]   show, validate, or edit config (validate, wizard, set, reset)\n"
                        "    /context        show current conversation context size\n"
                        "    /discuss        toggle discussion fallback (general knowledge)\n"
                        "    /embed [model]  show or switch embedding model\n"
                        "    /graph [sub]    GraphRAG status, finalize communities, or viz export\n"
                        "    /help [cmd]     show this help or details for a command\n"
                        "    /ingest <path>  ingest a file, directory, or glob\n"
                        "    /keys           show/set API keys (gemini, openai, brave, ollama_cloud)\n"
                        "    /list           list ingested documents\n"
                        "    /llm [opt val]  show or set LLM settings (temperature)\n"
                        "    /model [model]  show or switch LLM model\n"
                        "    /project [sub]  manage projects (list, new, switch, delete, folder)\n"
                        "    /pull <name>    pull an Ollama model\n"
                        "    /quit           exit Axon\n"
                        "    /rag [opt val]  show or set retrieval settings (topk, threshold, hybrid, …)\n"
                        "    /refresh        re-ingest documents whose content has changed\n"
                        "    /resume <id>    load a saved session\n"
                        "    /retry          retry the last query\n"
                        "    /search         toggle Brave web search fallback\n"
                        "    /sessions       list recent saved sessions\n"
                        "    /share [sub]    share projects (generate, redeem, revoke, list)\n"
                        "    /stale [days]   list documents not refreshed in N days (default: 7)\n"
                        "    /store [sub]    AxonStore multi-user mode (init, whoami)\n"
                        "    /theme [name]   show or switch markdown code-block highlight theme\n"
                        "    /vllm-url <url> set the vLLM server base URL\n"
                        "\n"
                        "    Shell:   !<cmd>  run a shell command (local-only default)\n"
                        "    Files:   @<file>  attach file context  ·  @<folder>/  attach all text files\n"
                        "\n"
                        "    /help <cmd>  for details  ·  e.g.  /help rag   /help share   /help project\n"
                        "    Tab  autocomplete  ·  ↑↓  history  ·  Ctrl+C  cancel  ·  Ctrl+D  exit\n"
                    )
            elif cmd == "/list":
                docs = brain.list_documents()
                if not docs:
                    print("    Knowledge base is empty.")
                else:
                    total = sum(d["chunks"] for d in docs)
                    print(f"\n    {len(docs)} file(s), {total} chunk(s)\n")
                    for d in docs:
                        print(f"    {d['source']:<60} {d['chunks']:>6}")
                    print()
            elif cmd == "/ingest":
                if not arg:
                    print("    Usage: /ingest <path|glob>  e.g. /ingest ./docs  /ingest ./src/*.py")
                else:
                    from axon.projects import ensure_project

                    # Prompt to create a project if none exist and currently in 'default'
                    if brain.should_recommend_project():
                        try:
                            print(
                                "\n    \033[1mNote\033[0m: You are about to ingest into the 'default' project."
                            )
                            print(
                                "    It is recommended to create a dedicated project to keep your data organized."
                            )
                            confirm = (
                                _read_input("    Create a new project now? [y/N]: ").strip().lower()
                            )
                            if confirm == "y":
                                new_name = _read_input("    New project name: ").strip().lower()
                                if new_name:
                                    try:
                                        ensure_project(new_name)
                                        brain.switch_project(new_name)
                                        print(f"    Switched to project '{new_name}'.\n")
                                    except ValueError as e:
                                        print(f"    {e}")
                        except (EOFError, KeyboardInterrupt):
                            print("\n    Cancelled project check.")
                    import glob as _glob

                    from axon.loaders import DirectoryLoader

                    # Expand glob pattern; fallback to literal path
                    matched = sorted(_glob.glob(arg, recursive=True))
                    if not matched:
                        # No glob match — try as plain directory
                        if Path(arg).is_dir():
                            matched = [arg]
                        else:
                            print(f"    No files matched: {arg}")
                    if matched:
                        loader_mgr = DirectoryLoader()
                        ingested, skipped = 0, 0
                        for path in matched:
                            p = Path(path)
                            if p.is_dir():
                                print(f"    {path} …", end="", flush=True)
                                asyncio.run(brain.load_directory(path))
                                print("    done")
                                ingested += 1
                            elif p.is_file():
                                ext = p.suffix.lower()
                                if ext in loader_mgr.loaders:
                                    brain.ingest(loader_mgr.loaders[ext].load(path))
                                    print(f"    {path}")
                                    ingested += 1
                                else:
                                    print(f"    !  Skipped (unsupported type): {path}")
                                    skipped += 1
                        print(f"    Done — {ingested} ingested, {skipped} skipped.")
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
                    print(f"    LLM:       {brain.config.llm_provider}/{brain.config.llm_model}")
                    print(
                        f"    Embedding: {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("    Usage:   /model <model>              (auto-detect provider)")
                    print("           /model <provider>/<model>   (switch provider too)")
                    print(f"    Providers: {', '.join(_PROVIDERS)}")
                    print(
                        f"    vLLM URL:  {brain.config.vllm_base_url}  (change with /vllm-url <url>)"
                    )
                elif "/" in arg:
                    provider, model = arg.split("/", 1)
                    if provider not in _PROVIDERS:
                        print(
                            f"    Unknown provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}"
                        )
                    else:
                        brain.config.llm_provider = provider
                        brain.config.llm_model = model
                        brain.llm = OpenLLM(brain.config)
                        print(f"    Switched LLM to {provider}/{model}")
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
                    print(f"    Switched LLM to {inferred}/{arg}")
                    _prompt_key_if_missing(inferred, brain)
            elif cmd == "/vllm-url":
                if not arg:
                    print(f"    Current vLLM base URL: {brain.config.vllm_base_url}")
                    print("    Usage: /vllm-url http://host:port/v1")
                else:
                    brain.config.vllm_base_url = arg
                    brain.llm._openai_clients = {}  # invalidate cached client
                    print(f"    vLLM base URL set to {arg}")
            elif cmd == "/embed":
                _EMBED_PROVIDERS = ("sentence_transformers", "ollama", "fastembed", "openai")
                if not arg:
                    print(
                        f"    Current:   {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("    Usage:   /embed <model>              (keep current provider)")
                    print("           /embed <provider>/<model>   (switch provider too)")
                    print(f"    Providers: {', '.join(_EMBED_PROVIDERS)}")
                    print("    Examples:")
                    print(
                        "      /embed all-MiniLM-L6-v2                    (sentence_transformers)"
                    )
                    print("      /embed /path/to/local/model                (local folder)")
                    print("      /embed ollama/nomic-embed-text")
                    print("      /embed fastembed/BAAI/bge-small-en")
                    print("    !  Changing embedding model invalidates existing indexed documents.")
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
                        print("    ⠿ Loading embedding model…", end="", flush=True)
                        brain.embedding = OpenEmbedding(brain.config)
                        print(
                            f"\r    Embedding switched to {brain.config.embedding_provider}/{brain.config.embedding_model}"
                        )
                        print("    Re-ingest your documents so they use the new embedding model.")
                    except Exception as e:
                        print(f"\r    Failed to load embedding: {e}")
            elif cmd == "/pull":
                if not arg:
                    print("    Usage: /pull <model-name>")
                else:
                    # use module-level os
                    if os.getenv("AXON_DRY_RUN"):
                        print("    Pulling models disabled in dry-run mode.")
                    else:
                        try:
                            import ollama as _ollama

                            print(f"    Pulling '{arg}' …")
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
                            print(f"\r    '{arg}' ready.{' ' * 50}")
                        except Exception as e:
                            print(f"    Pull failed: {e}")
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
                    print(f"    Graph visualization saved → {_out_path}")
                    print("    Open in your browser to explore the entity–relation graph.")
                except ImportError as _e:
                    print(f"    {_e}")
                except Exception as _e:
                    print(f"    Failed to export graph: {_e}")
            elif cmd == "/clear":
                _confirm = (
                    _read_input("    Clear knowledge base for the current project? [y/N]: ")
                    .strip()
                    .lower()
                )
                if _confirm not in ("y", "yes"):
                    print("    Clear cancelled.")
                    return False
                try:
                    brain._assert_write_allowed("clear")
                    clear_active_project(brain)
                    from axon import api as _api

                    project_key = getattr(brain, "_active_project", "default")
                    _api._source_hashes.pop(project_key, None)
                    if project_key == "default":
                        _api._source_hashes.pop("_global", None)
                    print(f"    Knowledge base cleared for project '{brain._active_project}'.")
                except PermissionError as _e:
                    print(f"    {_e}")
                except Exception as _e:
                    print(f"    Clear failed: {_e}")
            elif cmd == "/search":
                if brain.config.offline_mode:
                    print("    Offline mode is ON — web search is disabled.")
                elif brain.config.truth_grounding:
                    brain.config.truth_grounding = False
                    print("    Web search OFF — answers from local knowledge only.")
                else:
                    if not brain.config.brave_api_key:
                        print(
                            "    BRAVE_API_KEY is not set. Export it and restart, or set it with:"
                        )
                        print("     export BRAVE_API_KEY=your_key")
                    else:
                        brain.config.truth_grounding = True
                        print(
                            "    Web search ON — Brave Search will be used as fallback when local knowledge is insufficient."
                        )
            elif cmd == "/discuss":
                brain.config.discussion_fallback = not brain.config.discussion_fallback
                state = "ON" if brain.config.discussion_fallback else "OFF"
                print(f"    Discussion mode {state}.")
            elif cmd == "/rag":
                if not arg:
                    _grag_mode = getattr(brain.config, "graph_rag_mode", "local")
                    print(
                        f"\n    top-k           · {brain.config.top_k}\n"
                        f"    threshold       · {brain.config.similarity_threshold}\n"
                        f"    hybrid          · {'ON' if brain.config.hybrid_search else 'OFF'}\n"
                        f"    rerank          · {'ON' if brain.config.rerank else 'OFF'}"
                        + (f"  [{brain.config.reranker_model}]" if brain.config.rerank else "")
                        + "\n"
                        f"    hyde            · {'ON' if brain.config.hyde else 'OFF'}\n"
                        f"    multi-query     · {'ON' if brain.config.multi_query else 'OFF'}\n"
                        f"    step-back       · {'ON' if brain.config.step_back else 'OFF'}\n"
                        f"    decompose       · {'ON' if brain.config.query_decompose else 'OFF'}\n"
                        f"    compress        · {'ON' if brain.config.compress_context else 'OFF'}\n"
                        f"    sentence-window · {'ON' if getattr(brain.config, 'sentence_window', False) else 'OFF'}\n"
                        f"    crag-lite       · {'ON' if getattr(brain.config, 'crag_lite', False) else 'OFF'}\n"
                        f"    code-graph      · {'ON' if getattr(brain.config, 'code_graph', False) else 'OFF'}\n"
                        f"    raptor          · {'ON' if brain.config.raptor else 'OFF'}\n"
                        f"    graph-rag       · {'ON' if brain.config.graph_rag else 'OFF'}\n"
                        f"    graph-rag-mode  · {_grag_mode}\n"
                        f"\n    /help rag   for usage details\n"
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
                            print(f"    top-k set to {n}")
                        except Exception:
                            print("    Usage: /rag topk <integer 1–50>")
                    elif rag_opt == "threshold":
                        try:
                            v = float(rag_val)
                            assert 0.0 <= v <= 1.0
                            brain.config.similarity_threshold = v
                            print(f"    threshold set to {v}")
                        except Exception:
                            print("    Usage: /rag threshold <float 0.0–1.0>")
                    elif rag_opt == "hybrid":
                        brain.config.hybrid_search = not brain.config.hybrid_search
                        print(f"    Hybrid search {'ON' if brain.config.hybrid_search else 'OFF'}")
                    elif rag_opt == "rerank":
                        brain.config.rerank = not brain.config.rerank
                        print(f"    Reranker {'ON' if brain.config.rerank else 'OFF'}")
                    elif rag_opt == "hyde":
                        brain.config.hyde = not brain.config.hyde
                        print(f"    HyDE {'ON' if brain.config.hyde else 'OFF'}")
                    elif rag_opt == "multi":
                        brain.config.multi_query = not brain.config.multi_query
                        print(f"    Multi-query {'ON' if brain.config.multi_query else 'OFF'}")
                    elif rag_opt == "step-back":
                        brain.config.step_back = not brain.config.step_back
                        print(
                            f"    Step-back prompting {'ON' if brain.config.step_back else 'OFF'}"
                        )
                    elif rag_opt == "decompose":
                        brain.config.query_decompose = not brain.config.query_decompose
                        print(
                            f"    Query decomposition {'ON' if brain.config.query_decompose else 'OFF'}"
                        )
                    elif rag_opt == "compress":
                        brain.config.compress_context = not brain.config.compress_context
                        print(
                            f"    Context compression {'ON' if brain.config.compress_context else 'OFF'}"
                        )
                    elif rag_opt == "cite":
                        brain.config.cite = not brain.config.cite
                        print(f"    Inline citations {'ON' if brain.config.cite else 'OFF'}")
                    elif rag_opt == "raptor":
                        brain.config.raptor = not brain.config.raptor
                        print(
                            f"    RAPTOR hierarchical indexing {'ON' if brain.config.raptor else 'OFF'}"
                        )
                    elif rag_opt in ("graph-rag", "graph_rag", "graphrag"):
                        brain.config.graph_rag = not brain.config.graph_rag
                        print(
                            f"    GraphRAG entity retrieval {'ON' if brain.config.graph_rag else 'OFF'}"
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
                            print("    Usage: /rag sentence-window on|off")
                        else:
                            _on = (
                                rag_val.lower() in ("on", "true", "1")
                                if rag_val
                                else not getattr(brain.config, "sentence_window", False)
                            )
                            brain.config.sentence_window = _on
                            print(f"    Sentence-window retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("sentence-window-size", "sentence_window_size"):
                        try:
                            _sz = int(rag_val)
                            assert 1 <= _sz <= 10
                            brain.config.sentence_window_size = _sz
                            print(f"    Sentence-window size set to {_sz}")
                        except Exception:
                            print("    Usage: /rag sentence-window-size <integer 1–10>")
                    elif rag_opt in ("crag-lite", "crag_lite"):
                        _on = (
                            rag_val.lower() in ("on", "true", "1")
                            if rag_val
                            else not getattr(brain.config, "crag_lite", False)
                        )
                        brain.config.crag_lite = _on
                        print(f"    CRAG-lite corrective retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("code-graph", "code_graph"):
                        _on = (
                            rag_val.lower() in ("on", "true", "1")
                            if rag_val
                            else not getattr(brain.config, "code_graph", False)
                        )
                        brain.config.code_graph = _on
                        print(f"    Code-graph retrieval {'ON' if _on else 'OFF'}")
                    elif rag_opt in ("graph-rag-mode", "graph_rag_mode"):
                        _valid_modes = ("local", "global", "hybrid", "auto")
                        if rag_val.lower() not in _valid_modes:
                            print("    Usage: /rag graph-rag-mode local|global|hybrid|auto")
                        else:
                            brain.config.graph_rag_mode = rag_val.lower()
                            print(f"    GraphRAG mode set to '{rag_val.lower()}'")
                    elif rag_opt in ("max-hops", "max_hops", "graph-rag-max-hops"):
                        if not rag_val:
                            _cur = getattr(brain.config, "graph_rag_max_hops", 2)
                            print(f"    graph_rag_max_hops = {_cur}")
                            print("    Usage: /rag max-hops <int>  (0 = direct matches only)")
                        else:
                            try:
                                _hops = int(rag_val)
                                if _hops < 0:
                                    raise ValueError
                                brain.config.graph_rag_max_hops = _hops
                                print(f"    GraphRAG max hops set to {_hops}")
                            except ValueError:
                                print("    Usage: /rag max-hops <non-negative integer>")
                    elif rag_opt in ("hop-decay", "hop_decay", "graph-rag-hop-decay"):
                        if not rag_val:
                            _cur = getattr(brain.config, "graph_rag_hop_decay", 0.7)
                            print(f"    graph_rag_hop_decay = {_cur}")
                            print("    Usage: /rag hop-decay <float 0.0–1.0>")
                        else:
                            try:
                                _decay = float(rag_val)
                                if not (0.0 <= _decay <= 1.0):
                                    raise ValueError
                                brain.config.graph_rag_hop_decay = _decay
                                print(f"    GraphRAG hop decay set to {_decay}")
                            except ValueError:
                                print("    Usage: /rag hop-decay <float between 0.0 and 1.0>")
                    elif rag_opt in (
                        "distance-weighted",
                        "distance_weighted",
                        "graph-rag-distance-weighted",
                    ):
                        _choices = {
                            "on": True,
                            "true": True,
                            "1": True,
                            "off": False,
                            "false": False,
                            "0": False,
                        }
                        if rag_val.lower() not in _choices:
                            print("    Usage: /rag distance-weighted on|off")
                        else:
                            brain.config.graph_rag_distance_weighted = _choices[rag_val.lower()]
                            _state = (
                                "ON (Dijkstra)"
                                if brain.config.graph_rag_distance_weighted
                                else "OFF (BFS)"
                            )
                            print(f"    GraphRAG distance weighted → {_state}")
                    elif rag_opt == "rerank-model":
                        if not rag_val:
                            print(f"    Current reranker: {brain.config.reranker_model}")
                            print("    Usage: /rag rerank-model <HuggingFace ID or local path>")
                            print("    e.g.  /rag rerank-model BAAI/bge-reranker-base")
                            print("        /rag rerank-model ./models/bge-reranker-base")
                        else:
                            resolved = brain._resolve_model_path(rag_val)
                            if resolved != rag_val:
                                print(f"    Resolved to local path: {resolved}")
                            brain.config.reranker_model = resolved
                            brain.config.rerank = True  # auto-enable when setting a model
                            print(f"    Loading reranker '{resolved}'…")
                            try:
                                brain.reranker = OpenReranker(brain.config)
                                print(f"    Reranker → {resolved}  (rerank: ON)")
                            except Exception as e:
                                brain.config.rerank = False
                                print(f"    Failed to load reranker: {e}")
                    else:
                        print(
                            f"    Unknown option '{rag_opt}'. Try: topk, threshold, hybrid, rerank, rerank-model, "
                            f"hyde, multi, step-back, decompose, compress, cite, raptor, graph-rag, "
                            f"sentence-window, sentence-window-size, crag-lite, code-graph, graph-rag-mode, "
                            f"max-hops, hop-decay, distance-weighted"
                        )
            elif cmd == "/llm":
                if not arg:
                    print(
                        f"\n    temperature  · {brain.config.llm_temperature}\n"
                        f"    provider     · {brain.config.llm_provider}\n"
                        f"    model        · {brain.config.llm_model}\n"
                        f"\n    /llm temperature <0.0–2.0>   set generation temperature\n"
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
                            print(f"    Temperature set to {v}")
                        except Exception:
                            print("    Usage: /llm temperature <float 0.0–2.0>")
                    else:
                        print(f"    Unknown option '{llm_opt}'. Available: temperature")
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
                        print("    No projects yet. Use /project new <name> to create one.")
                    else:
                        print()
                        _print_project_tree(projects, active)
                    try:
                        from axon.projects import list_share_mounts as _list_mounts

                        _user_dir = Path(brain.config.projects_root)
                        _mounts = _list_mounts(_user_dir)
                        if _mounts:
                            print("\n    Mounted shares:")
                            for _m in _mounts:
                                _broken = "  [broken]" if _m.get("is_broken") else ""
                                print(f"      mounts/{_m['name']}  (owner: {_m['owner']}){_broken}")
                    except Exception:
                        pass
                    print(f"\n    Active: {active}")
                    print("    /project new <name>                      create + switch")
                    print("    /project new <parent>/<name>             create sub-project")
                    print("    /project switch <name>                   switch to existing")
                    print("    /project switch @projects|@mounts|@store switch to merged scope")
                    print("    /project switch mounts/<name>            switch to mounted share")
                    print(
                        "    /project refresh                         re-read mount version marker"
                    )
                    print("    /project folder                          open active project folder")
                    print(
                        "    /project rotate-keys [<name>]           rotate sealed DEK + invalidate shares\n"
                    )
                    print(
                        "    /project seal <name>                     encrypt at rest "
                        "(requires /store unlock)\n"
                    )
                elif sub == "new":
                    if not sub_arg:
                        print("    Usage: /project new <name>  [description]")
                        print("         /project new research/papers  (sub-project)")
                    else:
                        name_parts = sub_arg.split(maxsplit=1)
                        proj_name = name_parts[0].lower()
                        proj_desc = name_parts[1] if len(name_parts) > 1 else ""
                        try:
                            ensure_project(proj_name, proj_desc)
                            brain.switch_project(proj_name)
                            print(f"    Created and switched to project '{proj_name}'")
                            print(f"    {project_dir(proj_name)}")
                            print("    Use /ingest to add documents to this project.\n")
                        except ValueError as e:
                            print(f"    {e}")
                elif sub == "switch":
                    if not sub_arg:
                        print("    Usage: /project switch <name>")
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
                                    print(f"    Switched to project '{proj_name}'  [merged view]\n")
                                elif brain.vector_store.provider == "chroma":
                                    count = brain.vector_store.collection.count()
                                    print(
                                        f"    Switched to project '{proj_name}'  ({count} chunks)\n"
                                    )
                                else:
                                    print(f"    Switched to project '{proj_name}'\n")
                            except Exception as e:
                                print(f"    {e}")
                        else:
                            print(
                                f"    Project '{proj_name}' not found. Use /project list or /project new {proj_name}"
                            )
                elif sub == "refresh":
                    # Re-read the owner's version marker; reopen handles
                    # if newer. No-op when not on a mount.
                    try:
                        from axon.version_marker import MountSyncPendingError

                        refreshed = brain.refresh_mount()
                        marker = getattr(brain, "_mount_version_marker", None) or {}
                        if refreshed:
                            print(
                                f"    Refreshed mount: now at owner seq={marker.get('seq')}"
                                f" (generated_at={marker.get('generated_at')})\n"
                            )
                        elif brain._is_mounted_share():
                            print(f"    Mount already up to date (seq={marker.get('seq')}).\n")
                        else:
                            print(
                                "    /project refresh only applies to mounted shares "
                                "(mounts/<name>).\n"
                            )
                    except MountSyncPendingError as exc:
                        print(
                            f"    Owner has advanced but index files are still "
                            f"replicating: {exc}\n    Try again in a few seconds.\n"
                        )
                    except Exception as exc:
                        print(f"    Refresh failed: {exc}\n")
                elif sub == "delete":
                    if not sub_arg:
                        print("    Usage: /project delete <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        try:
                            confirm = (
                                _read_input(
                                    f"    !  Delete project '{proj_name}' and ALL its data? [y/N]: "
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
                                    print("    ↩️  Switched back to default project.")
                                delete_project(proj_name)
                                print(f"    Deleted project '{proj_name}'.\n")
                            except ProjectHasChildrenError as e:
                                print(f"    {e}")
                            except ValueError as e:
                                print(f"    {e}")
                        else:
                            print("    Cancelled.")
                elif sub == "folder":
                    active = brain._active_project
                    if active == "default":
                        print("    Default project uses config paths:")
                        print(f"      Vector store: {brain.config.vector_store_path}")
                        print(f"      BM25 index:   {brain.config.bm25_path}\n")
                    else:
                        folder = str(project_dir(active))
                        print(f"    {folder}")
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
                elif sub == "seal":
                    # Encrypt every content file in the project in place.
                    # Requires the [sealed] extra and an unlocked store.
                    if not sub_arg:
                        print("    Usage: /project seal <name>")
                    else:
                        from axon import security as _security

                        seal_name = sub_arg.strip().lower()
                        previously_active = brain._active_project
                        needs_switch_back = previously_active == seal_name
                        if needs_switch_back:
                            brain.switch_project("default")
                        try:
                            result = _security.project_seal(
                                seal_name,
                                Path(brain.config.projects_root),
                                config=brain.config,
                                embedding=brain.embedding,
                            )
                            status = result.get("status", "sealed")
                            files = result.get("files_sealed", 0)
                            print(f"    Project '{seal_name}': {status} ({files} files)")
                        except _security.SecurityError as exc:
                            print(f"    Seal failed: {exc}")
                        finally:
                            if needs_switch_back:
                                brain.switch_project(previously_active)
                elif sub == "rotate-keys":
                    proj = sub_arg.strip().lower() if sub_arg else brain._active_project
                    if not proj or proj == "default":
                        print("    Usage: /project rotate-keys <name>")
                        print(
                            "    Rotates the sealed project DEK and invalidates all existing shares."
                        )
                    else:
                        try:
                            from axon import security as _security

                            proj_root = _security.resolve_owned_sealed_project_path(
                                proj, Path(brain.config.projects_root)
                            )
                            result = _security.project_rotate_keys(proj_root)
                            resealed = result.get("files_resealed", 0)
                            revoked = result.get("invalidated_share_key_ids", [])
                            print(
                                f"    [rotate-keys] '{proj}': {resealed} file(s) re-encrypted, "
                                f"{len(revoked)} share(s) invalidated."
                            )
                            print("    Grantees must re-redeem a new share to regain access.")
                        except Exception as exc:
                            print(f"    rotate-keys failed: {exc}")
                else:
                    print(
                        f"    Unknown sub-command '{sub}'. "
                        "Try: list, new, switch, delete, folder, seal, rotate-keys"
                    )
            elif cmd == "/retry":
                if not _last_query:
                    print("    Nothing to retry — no previous query.")
                else:
                    user_input = _last_query
                    print(f"    ↩️  Retrying: {user_input}")
            elif cmd == "/context":
                _show_context(brain, chat_history, _last_sources, _last_query)
            elif cmd == "/sessions":
                _print_sessions(_list_sessions(project=brain._active_project))
            elif cmd == "/resume":
                if not arg:
                    print("    Usage: /resume <session-id>")
                else:
                    loaded = _load_session(arg, project=brain._active_project)
                    if loaded is None:
                        print(f"    Session '{arg}' not found. Use /sessions to list.")
                    else:
                        session = loaded
                        chat_history.clear()
                        chat_history.extend(session["history"])
                        turns = len(chat_history) // 2
                        print(f"    Loaded session {session['id']}  ({turns} turns)\n")
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
                        print("    Usage: /keys set <provider>")
                        print(f"    Providers: {', '.join(_provider_keys)}")
                    elif prov == "github_copilot":
                        print("    Starting GitHub OAuth device flow…")
                        try:
                            new_key = _copilot_device_flow()
                        except (EOFError, KeyboardInterrupt, RuntimeError) as e:
                            print(f"\n    Cancelled: {e}")
                        else:
                            env_name = _provider_keys[prov]
                            _save_env_key(env_name, new_key)
                            brain.config.copilot_pat = new_key
                            for k in ("_copilot", "_copilot_session", "_copilot_token"):
                                brain.llm._openai_clients.pop(k, None)
                            print(f"    {env_name} saved to {_env_file} and applied.")
                            print("    Switch provider: /model github_copilot/<model>")
                    else:
                        env_name = _provider_keys[prov]
                        try:
                            import getpass

                            new_key = getpass.getpass(f"  Enter {env_name} (hidden): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n    Cancelled.")
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
                                print(f"    {env_name} saved to {_env_file} and applied.")
                                print(f"    Switch provider: /model {prov}/<model-name>")
                            else:
                                print("    No key entered — nothing saved.")
                else:
                    print("\n    API Key Status\n    " + "─" * 50)
                    for prov, env_name in _provider_keys.items():
                        val = os.environ.get(env_name, "")
                        if val:
                            masked = val[:4] + "****" + val[-2:] if len(val) > 6 else "****"
                            status = f"set ({masked})"
                        else:
                            status = "not set"
                        print(f"    {prov:<14} {env_name:<22} {status}")
                    if _env_file.exists():
                        print(f"\n    Keys file: {_env_file}")
                    else:
                        print("\n    No keys file yet. Use /keys set <provider> to add keys.")
                    print("    /keys set <provider>  to set a key interactively")
                    print("    /help keys            for provider URLs and usage\n")
            elif cmd == "/share":
                # ── /share — project sharing lifecycle ──────────────────────────
                from axon import shares as _shares_mod

                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
                if not sub or sub == "list":
                    user_dir = Path(brain.config.projects_root)
                    _shares_mod.validate_received_shares(user_dir)
                    data = _shares_mod.list_shares(user_dir)
                    sharing = data.get("sharing", [])
                    shared = data.get("shared", [])
                    print("\n    Shares — issued by me:")
                    if sharing:
                        for s in sharing:
                            tag = " [revoked]" if s.get("revoked") else ""
                            print(f"      {s['project']} → {s['grantee']}  [ro]{tag}")
                    else:
                        print("      (none)")
                    print("\n    Shares — received:")
                    if shared:
                        for s in shared:
                            print(
                                f"      {s['owner']}/{s['project']} mounted as {s['mount']}  [ro]"
                            )
                    else:
                        print("      (none)")
                    print()
                elif sub == "generate":
                    # Usage: /share generate <project> <grantee> [--ttl-days N]
                    parts = sub_arg.split()
                    # Optional --ttl-days flag (issue #54)
                    ttl_days: int | None = None
                    if "--ttl-days" in parts:
                        try:
                            _idx = parts.index("--ttl-days")
                            ttl_days = int(parts[_idx + 1])
                            if ttl_days <= 0:
                                raise ValueError("ttl_days must be positive")
                            del parts[_idx : _idx + 2]
                        except (ValueError, IndexError):
                            print("    Invalid --ttl-days value (must be a positive integer).")
                            parts = []
                    if len(parts) < 2:
                        print("    Usage: /share generate <project> <grantee> [--ttl-days N]")
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
                                f"    Project '{proj}' not found. Use /project list to see projects."
                            )
                        else:
                            try:
                                try:
                                    from axon.security.seal import is_project_sealed as _is_sf

                                    _proj_sealed = _is_sf(proj_dir)
                                except ImportError:
                                    _proj_sealed = False
                                if _proj_sealed:
                                    import secrets as _secrets
                                    from datetime import datetime as _dt
                                    from datetime import timedelta as _td
                                    from datetime import timezone as _tz

                                    from axon import security as _security

                                    _key_id = f"ssk_{_secrets.token_hex(8)}"
                                    # v0.4.0: ttl_days is parsed earlier
                                    # in this block (search for "--ttl-days").
                                    # Convert to UTC datetime for the sealed
                                    # path's signed expiry sidecar.
                                    _expires_at = (
                                        _dt.now(_tz.utc) + _td(days=ttl_days) if ttl_days else None
                                    )
                                    result = _security.generate_sealed_share(
                                        owner_user_dir=user_dir,
                                        project=proj,
                                        grantee=grantee,
                                        key_id=_key_id,
                                        expires_at=_expires_at,
                                    )
                                    print(f"\n    Sealed share generated for project '{proj}'")
                                    print(f"    Grantee:      {grantee}")
                                    print("    Access:       read-only (encrypted-at-rest)")
                                    print(f"    Key ID:       {result['key_id']}")
                                    if result.get("expires_at"):
                                        print(f"    Expires at:   {result['expires_at']}")
                                    print(f"\n    Share string (send out-of-band to {grantee}):")
                                    print(f"\n      {result['share_string']}\n")
                                    print(
                                        f"    Revoke with:  /share revoke {result['key_id']} --project {proj}\n"
                                    )
                                else:
                                    result = _shares_mod.generate_share_key(
                                        owner_user_dir=user_dir,
                                        project=proj,
                                        grantee=grantee,
                                        ttl_days=ttl_days,
                                    )
                                    print(f"\n    Share key generated for project '{proj}'")
                                    print(f"    Grantee:      {grantee}")
                                    print("    Access:       read-only")
                                    print(f"    Key ID:       {result['key_id']}")
                                    if result.get("expires_at"):
                                        print(f"    Expires at:   {result['expires_at']}")
                                    print(f"\n    Share string (send this to {grantee}):")
                                    print(f"\n      {result['share_string']}\n")
                                    print(f"    Revoke with:  /share revoke {result['key_id']}\n")
                            except Exception as e:
                                print(f"    Share generation failed: {e}")
                elif sub == "redeem":
                    # Usage: /share redeem <share_string>
                    if not sub_arg:
                        print("    Usage: /share redeem <share_string>")
                    else:
                        user_dir = Path(brain.config.projects_root)
                        share_str = sub_arg.strip()
                        try:
                            import base64 as _b64

                            _padding = "=" * (-len(share_str) % 4)
                            _dec = _b64.urlsafe_b64decode(share_str + _padding).decode(
                                "utf-8", errors="replace"
                            )
                            from axon.security.share import is_sealed_share_envelope as _is_se

                            _is_sealed = _is_se(_dec)
                        except Exception:
                            _is_sealed = False
                        try:
                            if _is_sealed:
                                from axon import security as _security

                                result = _security.redeem_sealed_share(user_dir, share_str)
                                print("\n    Sealed share redeemed!")
                                print(f"    Project '{result['project']}' from {result['owner']}")
                                print(
                                    f"    Mounted at:  mounts/{result.get('mount_name', result['owner'] + '_' + result['project'])}"
                                )
                                print("    Access:      read-only (encrypted-at-rest)\n")
                            else:
                                result = _shares_mod.redeem_share_key(
                                    grantee_user_dir=user_dir,
                                    share_string=share_str,
                                )
                                print("\n    Share redeemed!")
                                print(f"    Project '{result['project']}' from {result['owner']}")
                                print(
                                    f"    Mounted at:  mounts/{result.get('mount_name', result['owner'] + '_' + result['project'])}"
                                )
                                print("    Access:      read-only\n")
                        except (ValueError, NotImplementedError) as e:
                            print(f"    Redeem failed: {e}")
                        except Exception as e:
                            print(f"    Redeem failed: {e}")
                elif sub == "revoke":
                    # Usage:
                    #   /share revoke <key_id>                              (legacy)
                    #   /share revoke <ssk_id> --project <name>             (sealed soft)
                    #   /share revoke <ssk_id> --project <name> --rotate    (sealed hard)
                    parts = sub_arg.split() if sub_arg else []
                    rotate = False
                    project_name: str | None = None
                    if "--rotate" in parts:
                        parts.remove("--rotate")
                        rotate = True
                    if "--project" in parts:
                        try:
                            _pidx = parts.index("--project")
                            project_name = parts[_pidx + 1]
                            del parts[_pidx : _pidx + 2]
                        except IndexError:
                            print(
                                "    --project requires a name: "
                                "/share revoke <key_id> --project <name> [--rotate]"
                            )
                            parts = []
                    if not parts:
                        print(
                            "    Usage: /share revoke <key_id>\n"
                            "         /share revoke <ssk_id> --project <name>            "
                            "(sealed soft)\n"
                            "         /share revoke <ssk_id> --project <name> --rotate   "
                            "(sealed hard)"
                        )
                    else:
                        key_id = parts[0].strip()
                        user_dir = Path(brain.config.projects_root)
                        if key_id.startswith("ssk_"):
                            from axon import security as _security

                            if not project_name:
                                print(
                                    "    Sealed-share revoke requires --project <name> "
                                    "to locate the wrap file."
                                )
                            else:
                                try:
                                    result = _security.revoke_sealed_share(
                                        owner_user_dir=user_dir,
                                        project=project_name,
                                        key_id=key_id,
                                        rotate=rotate,
                                    )
                                    if rotate:
                                        files = result.get("files_resealed", 0)
                                        invalid = result.get("invalidated_share_key_ids", [])
                                        print(
                                            f"    Sealed-share '{key_id}' hard-revoked: "
                                            f"DEK rotated, {files} files re-encrypted, "
                                            f"{len(invalid)} share(s) invalidated."
                                        )
                                        if invalid:
                                            print(
                                                f"    Re-issue these to surviving grantees: "
                                                f"{', '.join(invalid)}"
                                            )
                                    else:
                                        print(
                                            f"    Sealed-share '{key_id}' soft-revoked. "
                                            f"Note: cached DEKs on grantee machines remain "
                                            f"valid; pass --rotate for hard revoke."
                                        )
                                except _security.SecurityError as e:
                                    print(f"    Sealed revoke failed: {e}")
                        else:
                            # Legacy plaintext-mount share path.
                            try:
                                result = _shares_mod.revoke_share_key(
                                    owner_user_dir=user_dir,
                                    key_id=key_id,
                                )
                                print(f"    Share '{result['key_id']}' revoked.")
                            except ValueError as e:
                                print(f"    Revoke failed: {e}")
                            except Exception as e:
                                print(f"    Revoke failed: {e}")
                elif sub == "extend":
                    # Usage: /share extend <key_id> [--ttl-days N | --clear]
                    parts = sub_arg.split()
                    ttl_days: int | None = None
                    if "--clear" in parts:
                        parts.remove("--clear")
                    elif "--ttl-days" in parts:
                        try:
                            _idx = parts.index("--ttl-days")
                            ttl_days = int(parts[_idx + 1])
                            del parts[_idx : _idx + 2]
                        except (ValueError, IndexError):
                            print("    Invalid --ttl-days value (must be a positive integer).")
                            parts = []
                    if not parts:
                        print(
                            "    Usage: /share extend <key_id> [--ttl-days N | --clear]\n"
                            "      --ttl-days N : renew expiry to N days from now\n"
                            "      --clear      : remove expiry entirely"
                        )
                    else:
                        user_dir = Path(brain.config.projects_root)
                        try:
                            result = _shares_mod.extend_share_key(
                                owner_user_dir=user_dir,
                                key_id=parts[0].strip(),
                                ttl_days=ttl_days,
                            )
                            new_exp = result.get("expires_at") or "(no expiry)"
                            print(f"    Share '{result['key_id']}' expiry updated → {new_exp}")
                        except ValueError as e:
                            print(f"    Extend failed: {e}")
                        except Exception as e:
                            print(f"    Extend failed: {e}")
                else:
                    print(f"    Unknown sub-command '{sub}'.")
                    print(
                        "    Usage: /share list | generate <project> <grantee> [--ttl-days N] | "
                        "redeem <string> | revoke <key_id> | extend <key_id> [--ttl-days N | --clear]"
                    )
            elif cmd == "/store":
                # ── /store — AxonStore initialisation + identity ─────────────────
                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
                if not sub or sub == "whoami":
                    import getpass as _gp

                    username = _gp.getuser()
                    print(f"\n    User:       {username}")
                    print(f"    User dir:   {brain.config.projects_root}")
                    store_path = str(Path(brain.config.projects_root).parent)
                    print(f"    Store path: {store_path}")
                    print(f"    Project:    {brain._active_project}\n")
                elif sub == "init":
                    if not sub_arg:
                        print("    Usage: /store init <base_path>")
                        print("    Example: /store init ~/axon_data")
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
                            brain.config.projects_root = str(user_dir)
                            brain.config.vector_store_path = str(
                                user_dir / "default" / "vector_data"
                            )
                            brain.config.bm25_path = str(user_dir / "default" / "bm25_index")
                            try:
                                brain.config.save()
                            except Exception as _save_exc:
                                print(f"    Warning: could not save config: {_save_exc}")
                            print(f"\n    AxonStore initialised at {store_root}")
                            print(f"    Your directory:  {user_dir}")
                            print(f"    Username:        {username}")
                            print("    Use /share generate to share projects with others.\n")
                        except Exception as e:
                            print(f"    Store init failed: {e}")
                elif sub == "status":
                    from axon import security as _security

                    user_dir = Path(brain.config.projects_root)
                    try:
                        st = _security.store_status(user_dir)
                    except Exception as exc:
                        print(f"    Sealed-store status unavailable: {exc}")
                    else:
                        print(f"\n    Initialized:  {st.get('initialized', False)}")
                        print(f"    Unlocked:     {st.get('unlocked', False)}")
                        print(f"    Cipher suite: {st.get('cipher_suite', '') or '(none)'}\n")
                elif sub == "bootstrap":
                    from axon import security as _security

                    if not sub_arg:
                        print("    Usage: /store bootstrap <passphrase>")
                    else:
                        try:
                            _security.bootstrap_store(
                                Path(brain.config.projects_root), sub_arg.strip()
                            )
                            print("    Sealed-store bootstrapped and unlocked.")
                        except _security.SecurityError as exc:
                            print(f"    Bootstrap failed: {exc}")
                elif sub == "unlock":
                    from axon import security as _security

                    if not sub_arg:
                        print("    Usage: /store unlock <passphrase>")
                    else:
                        try:
                            _security.unlock_store(
                                Path(brain.config.projects_root), sub_arg.strip()
                            )
                            print("    Sealed-store unlocked.")
                        except _security.SecurityError as exc:
                            print(f"    Unlock failed: {exc}")
                elif sub == "lock":
                    from axon import security as _security

                    try:
                        _security.lock_store(Path(brain.config.projects_root))
                        print("    Sealed-store locked.")
                    except _security.SecurityError as exc:
                        print(f"    Lock failed: {exc}")
                elif sub == "change-passphrase":
                    from axon import security as _security

                    parts = sub_arg.split(maxsplit=1) if sub_arg else []
                    if len(parts) != 2:
                        print("    Usage: /store change-passphrase <old> <new>")
                    else:
                        try:
                            _security.change_passphrase(
                                Path(brain.config.projects_root),
                                parts[0].strip(),
                                parts[1].strip(),
                            )
                            print("    Sealed-store passphrase rotated.")
                        except _security.SecurityError as exc:
                            print(f"    Passphrase rotation failed: {exc}")
                else:
                    print(f"    Unknown sub-command '{sub}'.")
                    print(
                        "    Usage: /store whoami | /store init <base_path> | "
                        "/store status | /store bootstrap <pp> | /store unlock <pp> | "
                        "/store lock | /store change-passphrase <old> <new>"
                    )
            elif cmd == "/refresh":
                # ── /refresh — re-ingest changed documents ───────────────────────
                import hashlib as _hl_r

                from axon.loaders import DirectoryLoader as _DL

                versions = brain.get_doc_versions()
                if not versions:
                    print("    No tracked documents. Use /ingest to add documents.")
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
                    print("\n    Refresh complete:")
                    print(f"      Re-ingested: {len(reingested)}")
                    print(f"      Unchanged:   {len(skipped)}")
                    print(f"      Missing:     {len(missing)}")
                    if errors:
                        print(f"      Errors:      {len(errors)}")
                        for err in errors:
                            print(f"      {err}")
                    if reingested:
                        print("    Updated:")
                        for s in reingested:
                            print(f"      {s}")
                    print()
            elif cmd == "/stale":
                # ── /stale [days] — list documents not refreshed in N days ───────
                from datetime import datetime, timezone

                try:
                    threshold_days = int(arg) if arg.strip() else 7
                except ValueError:
                    print("    Usage: /stale [days]  (default: 7)")
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
                        print(f"\n    Stale documents (>{threshold_days} days):")
                        for age, src, _ts in stale:
                            print(f"      {age:6.1f}d  {src}")
                        print(f"\n    Total: {len(stale)}")
                        print("    Run /refresh to re-ingest changed documents.\n")
                    else:
                        print(f"    All documents are fresh (threshold: {threshold_days} days).")
            elif cmd == "/graph":
                # ── /graph — GraphRAG status + finalize ──────────────────────────
                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                if not sub or sub == "status":
                    # Prefer the backend's own status() for consistency across backends.
                    try:
                        _bs = brain._graph_backend.status()
                        _backend_name = _bs.get("backend", "graphrag")
                        _entities = _bs.get("entities", 0)
                        _relations = _bs.get("relations", 0)
                        _summaries = _bs.get("community_summaries", 0)
                        _communities = _bs.get("communities", 0)
                    except Exception:
                        # Graceful fallback for legacy or stub backends.
                        _backend_name = "graphrag"
                        _entities = len(getattr(brain, "_entity_graph", {}) or {})
                        _relations = sum(
                            len(v) for v in (getattr(brain, "_relation_graph", {}) or {}).values()
                        )
                        _summaries = len(getattr(brain, "_community_summaries", {}) or {})
                        _communities = 0
                    in_progress = getattr(brain, "_community_build_in_progress", False)
                    print("\n    GraphRAG status:")
                    print(f"      Backend:               {_backend_name}")
                    print(f"      Entities:              {_entities}")
                    print(f"      Relation edges:        {_relations}")
                    print(f"      Communities:           {_communities}")
                    print(f"      Community summaries:   {_summaries}")
                    print(
                        f"      Community build:       {'in progress' if in_progress else 'idle'}"
                    )
                    print(f"      graph_rag enabled:     {brain.config.graph_rag}")
                    print()
                elif sub == "finalize":
                    if not brain.config.graph_rag:
                        print("    GraphRAG is disabled. Enable with /rag graph-rag first.")
                    else:
                        print("    Finalizing graph communities… (this may take a moment)")
                        try:
                            backend = getattr(brain, "_graph_backend", None)
                            if backend is not None and callable(getattr(backend, "finalize", None)):
                                _r = backend.finalize(force=True)
                                _stat = getattr(_r, "status", "ok")
                                _det = getattr(_r, "detail", "")
                                if _stat == "not_applicable":
                                    print(
                                        f"    Backend has no finalize step — {_det or 'no community detection on this backend'}."
                                    )
                                else:
                                    summaries = getattr(brain, "_community_summaries", {}) or {}
                                    print(
                                        f"    Done. {len(summaries)} community summaries generated.\n"
                                    )
                            else:
                                brain.finalize_graph()
                                summaries = getattr(brain, "_community_summaries", {}) or {}
                                print(
                                    f"    Done. {len(summaries)} community summaries generated.\n"
                                )
                        except Exception as e:
                            print(f"    Finalize failed: {e}")
                elif sub == "conflicts":
                    backend = getattr(brain, "_graph_backend", None)
                    fn = getattr(backend, "list_conflicts", None) if backend else None
                    if not callable(fn):
                        _bid = getattr(backend, "BACKEND_ID", "none") if backend else "none"
                        print(f"    Backend '{_bid}' does not track conflicted facts.")
                    else:
                        try:
                            _rows = fn(limit=100)
                        except Exception as e:
                            print(f"    Listing conflicts failed: {e}")
                            _rows = []
                        if not _rows:
                            print("    No conflicted facts.")
                        else:
                            print(f"\n    {len(_rows)} conflicted fact(s):")
                            for _row in _rows[:50]:
                                _va = _row.get("valid_at", "")
                                print(
                                    f"      [{_va[:10] if _va else '?'}] "
                                    f"{_row.get('subject','?')} "
                                    f"{_row.get('relation','?')} "
                                    f"{_row.get('object','?')}"
                                )
                            print()
                elif sub == "retrieve":
                    sub_parts2 = arg.split(maxsplit=1)
                    if len(sub_parts2) < 2 or not sub_parts2[1].strip():
                        print("    Usage: /graph retrieve <query> [--at ISO-TIMESTAMP] [--top-k N]")
                    else:
                        from datetime import datetime as _dt

                        from axon.graph_backends.base import RetrievalConfig as _RCfg

                        _raw = sub_parts2[1]
                        # Parse trailing flags off the end.
                        _at: _dt | None = None
                        _topk = 10
                        _tokens = _raw.split()
                        _qry_tokens: list[str] = []
                        i = 0
                        while i < len(_tokens):
                            t = _tokens[i]
                            if t == "--at" and i + 1 < len(_tokens):
                                try:
                                    _at = _dt.fromisoformat(_tokens[i + 1].replace("Z", "+00:00"))
                                except ValueError:
                                    print(f"    Bad --at value: {_tokens[i + 1]}")
                                    _at = None
                                i += 2
                            elif t == "--top-k" and i + 1 < len(_tokens):
                                try:
                                    _topk = int(_tokens[i + 1])
                                except ValueError:
                                    print(f"    Bad --top-k value: {_tokens[i + 1]}")
                                i += 2
                            else:
                                _qry_tokens.append(t)
                                i += 1
                        _q = " ".join(_qry_tokens).strip()
                        backend = getattr(brain, "_graph_backend", None)
                        if backend is None or not callable(getattr(backend, "retrieve", None)):
                            print("    No graph backend is active.")
                        elif not _q:
                            print("    Query string is empty after stripping flags.")
                        else:
                            try:
                                _ctxs = backend.retrieve(
                                    _q,
                                    _RCfg(top_k=_topk, point_in_time=_at),
                                    None,
                                )
                            except Exception as e:
                                print(f"    Retrieve failed: {e}")
                                _ctxs = []
                            if not _ctxs:
                                print("    No graph contexts matched.")
                            else:
                                print(
                                    f"\n    {len(_ctxs)} context(s)"
                                    + (f" at {_at.isoformat()}" if _at else "")
                                    + ":"
                                )
                                for _c in _ctxs[:20]:
                                    _txt = (_c.text or "")[:120].replace("\n", " ")
                                    print(f"      [{_c.score:.3f} {_c.context_type}] {_txt}")
                                print()
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
                        print(f"    Graph saved to: {out}")
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
                                "    Open the file in your browser to explore the entity–relation graph."
                            )
                    except Exception as e:
                        print(f"    Graph visualisation failed: {e}")
                else:
                    print(f"    Unknown sub-command '{sub}'.")
                    print(
                        "    Usage: /graph status | /graph finalize | /graph conflicts | "
                        "/graph retrieve <query> [--at TS] [--top-k N] | /graph viz [path]"
                    )
            elif cmd == "/theme":
                global _MD_CODE_THEME
                _theme_arg = arg.strip()
                if not _theme_arg or _theme_arg == "show":
                    print(f"    Current markdown code theme: {_MD_CODE_THEME}")
                    print("    Use /theme list  to see popular themes.")
                    print("    Use /theme <name>  to switch.")
                elif _theme_arg == "list":
                    _popular = [
                        "monokai",
                        "dracula",
                        "gruvbox-dark",
                        "nord",
                        "one-dark",
                        "solarized-dark",
                        "solarized-light",
                        "github-dark",
                        "vim",
                        "vs",
                        "friendly",
                        "tango",
                    ]
                    print("    Popular themes (any Pygments style name is accepted):")
                    for _t in _popular:
                        marker = " ◀ current" if _t == _MD_CODE_THEME else ""
                        print(f"      {_t}{marker}")
                else:
                    try:
                        from pygments.styles import get_style_by_name

                        get_style_by_name(_theme_arg)  # raises if unknown
                        _MD_CODE_THEME = _theme_arg
                        _save_pref("md_code_theme", _theme_arg)
                        print(f"    Markdown code theme set to: {_theme_arg}")
                    except Exception:
                        print(f"    Unknown theme '{_theme_arg}'. Run /theme list for suggestions.")
            elif cmd == "/config":
                _cfg_path = (
                    getattr(brain.config, "_loaded_path", "") if brain is not None else ""
                ) or ""
                _handle_config_cmd(arg.strip(), brain, _cfg_path)
            elif cmd == "/agent":
                _agent_mode = not _agent_mode
                state = "ON" if _agent_mode else "OFF"
                print(
                    f"    Agent mode {state}. LLM can now call Axon tools directly."
                    if _agent_mode
                    else "    Agent mode OFF. Back to Q&A mode."
                )
            elif cmd == "/debug":
                import logging as _logging

                _NOISY_LOGGERS = (
                    "httpcore",
                    "httpcore.http11",
                    "hpack",
                    "urllib3",
                    "markdown_it",
                    "asyncio",
                    "google_genai",
                    "google.genai",
                )
                # Determine current state from the first logger's level
                _first = _logging.getLogger(_NOISY_LOGGERS[0])
                _debug_on = _first.level != _logging.DEBUG
                _new_level = _logging.DEBUG if _debug_on else _logging.WARNING
                for _nl in _NOISY_LOGGERS:
                    _logging.getLogger(_nl).setLevel(_new_level)
                if _debug_on:
                    print("    Debug logging ON — verbose library logs enabled.")
                else:
                    print("    Debug logging OFF.")
            elif cmd == "/governance":
                # ── /governance — operator status and audit log ──────────────────
                import json as _json_gov

                sub_parts = arg.split(maxsplit=1) if arg else []
                sub = sub_parts[0].lower() if sub_parts else "overview"
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""
                _gov_host = getattr(brain.config, "api_host", "127.0.0.1") or "127.0.0.1"
                _gov_base = f"http://{_gov_host}:{getattr(brain.config, 'api_port', 8000)}"
                try:
                    import urllib.parse as _up_gov
                    import urllib.request as _ur_gov

                    if sub in ("", "overview"):
                        with _ur_gov.urlopen(f"{_gov_base}/governance/overview", timeout=10) as _gr:
                            _gd = _json_gov.loads(_gr.read())
                        print("\n    === Governance Overview ===")
                        print(_json_gov.dumps(_gd, indent=4))
                    elif sub == "audit":
                        _qs = _up_gov.urlencode(
                            {"limit": 20, "action": sub_arg} if sub_arg else {"limit": 20}
                        )
                        _gurl = f"{_gov_base}/governance/audit?{_qs}"
                        with _ur_gov.urlopen(_gurl, timeout=10) as _gr:
                            _gd = _json_gov.loads(_gr.read())
                        events = _gd.get("events", [])
                        print(f"\n    === Audit Log ({len(events)} entries) ===")
                        for _ev in events:
                            _ts = _ev.get("timestamp", "")[:19]
                            _act = _ev.get("action", "")
                            _proj = _ev.get("project", "")
                            _stat = _ev.get("status", "")
                            print(f"    {_ts}  [{_stat:>7}]  {_act:<30}  {_proj}")
                    elif sub == "sessions":
                        with _ur_gov.urlopen(
                            f"{_gov_base}/governance/copilot/sessions", timeout=10
                        ) as _gr:
                            _gd = _json_gov.loads(_gr.read())
                        print(_json_gov.dumps(_gd, indent=4))
                    elif sub == "projects":
                        with _ur_gov.urlopen(f"{_gov_base}/governance/projects", timeout=10) as _gr:
                            _gd = _json_gov.loads(_gr.read())
                        projects_data = _gd.get("projects", [])
                        print(f"\n    === Governance Projects ({len(projects_data)}) ===")
                        for _p in projects_data:
                            _mn = _p.get("maintenance", {}).get("maintenance_state", "normal")
                            _lc = _p.get("maintenance", {}).get("active_leases", 0)
                            print(
                                f"    {_p.get('name', ''):<30}  " f"state={_mn:<12}  leases={_lc}"
                            )
                    elif sub == "graph-rebuild":
                        _req_gov = _ur_gov.Request(
                            f"{_gov_base}/governance/graph/rebuild",
                            data=b"{}",
                            method="POST",
                            headers={"Content-Type": "application/json"},
                        )
                        with _ur_gov.urlopen(_req_gov, timeout=60) as _gr:
                            _gd = _json_gov.loads(_gr.read())
                        print(f"    Graph rebuild triggered: {_gd}")
                    else:
                        print(
                            f"    Unknown sub-command '{sub}'.\n"
                            "    Usage: /governance [overview | audit [action] | "
                            "sessions | projects | graph-rebuild]"
                        )
                except Exception as _ge:
                    print(
                        f"    Governance command failed: {_ge}\n"
                        "    Make sure the Axon API server is running (axon-api)."
                    )
            elif cmd == "/mount-refresh":
                # ── /mount-refresh — pull owner's latest for an active mount ────
                try:
                    refreshed = brain.refresh_mount()
                    _proj_name = getattr(brain, "_active_project", "unknown")
                    if refreshed:
                        print(f"    Mount '{_proj_name}' refreshed — new owner version detected.")
                    else:
                        print(f"    Mount '{_proj_name}' is up to date (no new version).")
                except Exception as _mre:
                    print(f"    Mount refresh failed: {_mre}")
            else:
                print(f"    Unknown command: {cmd}. Type /help for options.")
            if cmd != "/retry":
                return False
        # --- @file expansion: replace @path references with file contents ---
        query_text = _expand_at_files(user_input)

        # Always resolve @file references — even when _expand_at_files could not
        # expand the content (file not at CWD).  The resolved absolute path is
        # prepended for agent mode regardless so the LLM can call ingest_path
        # with the correct path instead of the user-typed relative form.
        def _resolve_at_path(p: str) -> str:
            expanded = Path(p).expanduser()
            if expanded.is_absolute():
                logger.debug("@file resolved (absolute): %s", expanded)
                return str(expanded)
            cwd_resolved = expanded.resolve()
            if cwd_resolved.exists():
                logger.debug("@file resolved (cwd): %s", cwd_resolved)
                return str(cwd_resolved)
            home_resolved = Path.home() / p
            if home_resolved.exists():
                logger.debug("@file resolved (home fallback): %s", home_resolved)
                return str(home_resolved)
            logger.warning(
                "@file '%s' not found — tried cwd (%s) and home (%s). "
                "Use ~/path or an absolute path to be explicit.",
                p,
                cwd_resolved,
                home_resolved,
            )
            return str(home_resolved)  # best guess pointing at home even if missing

        at_files = re.findall(r"(?<!\S)@(\S+)", user_input)
        if at_files:
            at_files_abs = [_resolve_at_path(p) for p in at_files]
            print(f"    Attached: {', '.join(at_files_abs)}")
            # In agent mode, prepend the resolved absolute file paths so the agent can
            # call ingest_path(path=...) instead of add_text with the content blob.
            if _agent_mode:
                paths_note = "  ".join(at_files_abs)
                query_text = f"[Attached file path(s): {paths_note}]\n\n{query_text}"
        # --- Regular query — use Rich Console for output + toolbar spinner ---
        response_parts: list = []
        _cancelled = False
        try:
            from rich.console import Console as _RC

            _console = _RC(file=sys.stdout, force_terminal=True)
            # Helpers that route Rich output through prompt_toolkit's own output
            # pipeline so ANSI codes render correctly on Windows (no ?[ artifacts).
            import io as _io

            def _rich_print(markup: str, **kw) -> None:
                _buf = _io.StringIO()
                _cap = _RC(file=_buf, force_terminal=True, width=_console.width or 120)
                _cap.print(markup, **kw)
                _ansi = _buf.getvalue()
                sys.stdout.write(_ansi)
                sys.stdout.flush()

            def _format_user_label(text: str, tc: int) -> str:
                """Wrap user input so every line gets the full-width grey15 highlight."""
                prefix = "    > "
                cont = "      "  # same width, no marker
                avail = max(tc - len(prefix), 20)
                lines = textwrap.wrap(text, width=avail) or [""]
                parts = [f"{prefix}{lines[0]}".ljust(tc)]
                for ln in lines[1:]:
                    parts.append(f"{cont}{ln}".ljust(tc))
                return "\n".join(parts)

            def _rich_render(text: str, indent: str = "", right_margin: int = 0) -> None:
                _buf = _io.StringIO()
                _w = max(40, (int(_console.width or 120)) - len(indent) - right_margin)
                _cap = _RC(file=_buf, force_terminal=True, width=_w)
                _cap.print(_make_math_renderable(text))
                _ansi = _buf.getvalue()
                if indent:
                    _ansi = "\n".join(indent + ln if ln else ln for ln in _ansi.split("\n"))
                sys.stdout.write(_ansi)
                sys.stdout.flush()

            if stream:
                if _agent_mode:
                    # Agent mode: run_agent_loop in a thread with spinner feedback,
                    # then render the result as rich markdown (same as non-streaming path).
                    from axon.agent import REPL_TOOLS, run_agent_loop

                    def _confirm_cb(msg: str) -> bool:
                        # Called from the agent background thread — must suspend the
                        # prompt_toolkit Application before reading stdin, otherwise
                        # keystrokes go to the Application rather than to input().
                        # NOTE: `run_in_terminal` (module-level) calls ensure_future()
                        # synchronously in the calling thread → fails with "no current
                        # event loop".  Use `in_terminal` (async ctx-mgr) inside a
                        # proper coroutine scheduled on the REPL event loop instead.
                        try:
                            _result: list = []

                            def _do_confirm() -> None:
                                ans = input(f"\n  ⚠️  {msg} [y/N]: ").strip().lower()
                                _result.append(ans == "y")

                            if _pt_app is not None and _repl_event_loop is not None:
                                import asyncio as _aio

                                from prompt_toolkit.application import (
                                    in_terminal as _in_terminal,
                                )

                                async def _run_confirm() -> None:
                                    async with _in_terminal():
                                        _do_confirm()

                                fut = _aio.run_coroutine_threadsafe(
                                    _run_confirm(),
                                    _repl_event_loop,
                                )
                                fut.result()
                                return _result[0] if _result else False
                            else:
                                _do_confirm()
                                return _result[0] if _result else False
                        except (EOFError, KeyboardInterrupt):
                            return False

                    # Tool-step callback: print each Axon tool's result as a
                    # labelled block immediately after execution.
                    # run_shell prints its own "Bash" block — skip it here.
                    _SKIP_STEP_CB = {"run_shell"}
                    # Tools whose output is retrieval/listing detail — show only
                    # the first (summary) line; the synthesised answer carries the value.
                    _SUMMARY_ONLY_TOOLS = {
                        "search_knowledge",
                        "query_knowledge",
                        "list_knowledge",
                        "graph_data",
                    }
                    # Collect tool steps during agent execution; render them
                    # inside the Axon response block after the agent finishes.
                    _tool_steps: list[tuple[str, str]] = []

                    def _agent_step_cb(tool_name: str, result: str) -> None:
                        if tool_name in _SKIP_STEP_CB:
                            return
                        _tool_steps.append((tool_name, result))

                    _agent_result: list = []
                    _agent_err: list = []
                    _agent_spin_stop = threading.Event()

                    def _run_agent() -> None:
                        try:
                            _agent_result.append(
                                run_agent_loop(
                                    brain.llm,
                                    brain,
                                    query_text,
                                    chat_history,
                                    tools=REPL_TOOLS,
                                    confirm_cb=_confirm_cb,
                                    step_cb=_agent_step_cb,
                                )
                            )
                        except Exception as _ae:
                            _agent_err.append(_ae)
                        finally:
                            _agent_spin_stop.set()

                    _agent_thread = threading.Thread(target=_run_agent, daemon=True)
                    _agent_thread.start()
                    if not quiet:
                        # Echo user message with full-width highlight.
                        _tc_ag = shutil.get_terminal_size().columns
                        _rich_print(
                            f"\n[bold white on grey15]{_format_user_label(user_input, _tc_ag)}[/bold white on grey15]\n"
                        )
                        _spin_state["active"] = True
                        _spin_state["idx"] = 0
                        _ast = threading.Thread(target=_animate_spinner, daemon=True)
                        _ast.start()
                        _agent_spin_stop.wait()
                        _spin_state["active"] = False
                        _ast.join(timeout=0.5)
                        if _pt_app is not None:
                            try:
                                _pt_app.invalidate()
                            except Exception:
                                pass
                    else:
                        _agent_thread.join()
                    if _agent_err:
                        _rich_print(f"    [bold red]✦[/bold red] ⚠️  {_agent_err[0]}\n")
                        response_parts = []
                    else:
                        _agent_response = _agent_result[0] if _agent_result else ""
                        _rich_print("    [bold yellow]✦[/bold yellow]")
                        _YEL = "\x1b[33m"
                        _GRN = "\x1b[1;32m"
                        _RED = "\x1b[1;31m"
                        _DIM = "\x1b[2m"
                        _RST = "\x1b[0m"
                        for _tname, _tresult in _tool_steps:
                            _is_err = (
                                _tresult.startswith("⚠️")
                                or _tresult.startswith("🚫")
                                or _tresult.startswith("No files matched")
                                or _tresult.startswith("No path")
                                or _tresult.startswith("No text")
                                or _tresult.startswith("Unknown tool")
                            )
                            _icon = "✗" if _is_err else "✓"
                            _clr = _RED if _is_err else _GRN
                            _tlines = _tresult.strip().splitlines()
                            _first = _tlines[0] if _tlines else "(no output)"
                            _rest = _tlines[1:]
                            print(f"    {_DIM}↳ {_tname}{_RST}  " f"{_clr}{_icon}{_RST}  {_first}")
                            if _is_err:
                                for _ln in _rest[:3]:
                                    print(f"           {_ln}")
                            elif _tname in _SUMMARY_ONLY_TOOLS:
                                if _rest:
                                    print(
                                        f"           {_DIM}({len(_rest)} line(s) collapsed){_RST}"
                                    )
                            else:
                                for _ln in _rest[:3]:
                                    print(f"           {_ln}")
                                if len(_rest) > 3:
                                    print(f"           {_DIM}… +{len(_rest) - 3} more{_RST}")
                        if _tool_steps:
                            print()
                        _rich_render(_agent_response, indent="    ", right_margin=4)
                        print()
                        response_parts = [_agent_response]
                    response = "".join(response_parts)
                    if not _cancelled:
                        chat_history.append({"role": "user", "content": user_input})
                        chat_history.append({"role": "assistant", "content": response})
                        _last_query = user_input
                        _save_session(session)
                    return False
                else:
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
                            _rich_print(f"[dim]    {'  │  '.join(changes)}[/dim]")
                        _last_config_snapshot = _snap
                    # Echo user message then show thinking indicator in conversation area.
                    if not quiet:
                        _tc_in = shutil.get_terminal_size().columns
                        _rich_print(
                            f"\n[bold white on grey15]{_format_user_label(user_input, _tc_in)}[/bold white on grey15]\n"
                        )
                    # Collect all streaming tokens while spinner shows.
                    # Writing token-by-token conflicts with prompt_toolkit's
                    # Application redraws (invalidate races run_in_terminal),
                    # causing truncated output. Collect first, render once after
                    # spinner fully stops. Rich markdown is also restored this way.
                    _spin_state["active"] = True
                    _spin_state["idx"] = 0
                    _st = threading.Thread(target=_animate_spinner, daemon=True)
                    _st.start()
                    _stream_error = None
                    try:
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            response_parts.append(chunk)
                    except KeyboardInterrupt:
                        _cancelled = True
                    except Exception as _se:
                        _stream_error = _se
                    finally:
                        _spin_state["active"] = False
                        _st.join(timeout=0.3)
                        if _pt_app is not None:
                            try:
                                _pt_app.invalidate()
                            except Exception:
                                pass
                    _accumulated = "".join(response_parts)
                    if _stream_error is not None:
                        _rich_print(f"    [bold red]✦[/bold red] ⚠️  {_stream_error}\n")
                    elif _accumulated:
                        _rich_print("    [bold yellow]✦[/bold yellow]")
                        _rich_render(_accumulated, indent="    ", right_margin=4)
                        print()
                    else:
                        _rich_print("    [bold yellow]✦[/bold yellow] (no response)\n")
                    if _cancelled:
                        _rich_print("    ⚠  Cancelled.\n")
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
                            _rich_print(f"[dim]    {'  │  '.join(changes)}[/dim]")
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
                    # Echo user message with full-width highlight.
                    _tc_in2 = shutil.get_terminal_size().columns
                    _rich_print(
                        f"\n[bold white on grey15]{_format_user_label(user_input, _tc_in2)}[/bold white on grey15]\n"
                    )
                    _spin_state["active"] = True
                    _spin_state["idx"] = 0
                    _st2 = threading.Thread(target=_animate_spinner, daemon=True)
                    _st2.start()
                    _spin_stop2.wait()
                    _spin_state["active"] = False
                    _st2.join(timeout=0.5)
                    if _pt_app is not None:
                        try:
                            _pt_app.invalidate()
                        except Exception:
                            pass
                else:
                    _qt.join()
                if _err:
                    raise _err[0]
                response = _result[0] if _result else ""
                _rich_print("    [bold yellow]✦[/bold yellow]")
                _rich_render(response, indent="    ", right_margin=4)
                print()
                response_parts = [response]
        except ImportError:
            # rich not available — plain fallback
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx = [0]

            def _spin_plain() -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    sys.stdout.write(f"\r    ✦ {f} thinking…")
                    sys.stdout.flush()
                    _spin_idx[0] += 1

            if not quiet:
                print()
                _spt = threading.Thread(target=_spin_plain, daemon=True)
                _spt.start()
            response = brain.query(query_text, chat_history=chat_history)
            if not quiet:
                _spin_stop.set()
            print(f"\n\033[1;33m✦\033[0m {response}\n")
            response_parts = [response]
        response = "".join(response_parts)
        if not _cancelled:
            # Append both turns so future queries have full context
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            _last_query = user_input
            _save_session(session)  # persist after every turn
        return False

    # ── Main REPL runner ───────────────────────────────────────────────────────
    # Always define _animate_spinner so _process_input_sync can reference it
    # whether running in Application mode or readline fallback.
    def _animate_spinner() -> None:
        """Animate the spinner in the toolbar at ~12fps via prompt_toolkit invalidate."""
        while _spin_state["active"]:
            _spin_state["idx"] += 1
            if _pt_app is not None:
                try:
                    _pt_app.invalidate()
                except Exception:
                    pass
            time.sleep(0.08)

    if _pt_app is not None and _scripted_inputs is None:
        # Application loop: runs continuously so toolbar+input are always
        # visible.  Processing happens in asyncio.to_thread so the event loop
        # (and the Application) keep running during LLM calls.
        async def _repl_loop_async() -> None:
            nonlocal _input_queue, _repl_event_loop
            _input_queue = asyncio.Queue()
            _repl_event_loop = asyncio.get_running_loop()
            # patch_stdout must be entered after the event loop is running
            # so it binds to the correct loop for run_in_terminal calls.
            with _pt_patch_stdout(raw=True):
                app_task = asyncio.ensure_future(_pt_app.run_async())

                def _on_app_done(fut: asyncio.Future) -> None:
                    exc = fut.exception() if not fut.cancelled() else None
                    if exc and not isinstance(exc, KeyboardInterrupt | EOFError | SystemExit):
                        import logging as _lg2

                        _lg2.getLogger("Axon").warning("REPL Application exited with: %s", exc)
                    _input_queue.put_nowait(None)

                app_task.add_done_callback(_on_app_done)
                try:
                    while True:
                        text = await _input_queue.get()
                        if text is None:
                            break
                        text = text.strip()
                        if not text:
                            continue
                        should_exit = await asyncio.to_thread(lambda t=text: _process_input_sync(t))
                        if should_exit:
                            break
                except (KeyboardInterrupt, EOFError):
                    pass
                finally:
                    try:
                        _pt_app.exit()
                    except Exception:
                        pass  # Application may have already stopped
                    try:
                        await app_task
                    except Exception:
                        pass

        asyncio.run(_repl_loop_async())
    else:
        # Readline / plain fallback loop
        while True:
            try:
                user_input = _read_input().strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            should_exit = False
            try:
                should_exit = _process_input_sync(user_input)
            finally:
                # In non-Application mode, reprint the status bar to keep the UI informative.
                if _pt_app is None:
                    try:
                        _print_status_bar()
                    except Exception:
                        pass
            if should_exit:
                break
