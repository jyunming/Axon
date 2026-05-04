"""EFF large-wordlist passphrase generation for sealed-store unlock.

Bundles the EFF's Diceware large wordlist (7,776 words, CC BY 3.0 US —
see ``data/LICENSE-EFF-WORDLIST.txt``) and exposes
:func:`generate_passphrase` for CLI / REPL / REST / MCP surfaces.

Six words ≈ 77 bits of entropy (``log2(7776**6)``), enough that scrypt
``N=2**15`` brute force is infeasible. The default of 6 was picked to
match the EFF's own recommendation.
"""
from __future__ import annotations

import secrets
from functools import lru_cache
from importlib import resources
from typing import Final

_MIN_WORDS: Final = 4
_MAX_WORDS: Final = 12
# Use a space by default: 4 EFF entries are themselves hyphenated
# (drop-down, felt-tip, t-shirt, yo-yo) so "-" as separator is visually
# ambiguous. The passphrase is treated as opaque by Axon's scrypt KDF —
# the separator only affects how the user reads / types it.
_DEFAULT_SEPARATOR: Final = " "


@lru_cache(maxsize=1)
def _load_wordlist() -> tuple[str, ...]:
    """Read and cache the EFF wordlist.

    The file format is ``<5-digit-dice-roll>\\t<word>\\n``; we strip the
    leading dice column and return the words in source order. Result is
    cached so repeated calls don't re-read the package resource.
    """
    raw = (
        resources.files("axon.security.data")
        .joinpath("eff_large_wordlist.txt")
        .read_text(encoding="utf-8")
    )
    words: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # "<dice>\t<word>" — split on whitespace, take last column
        parts = line.split()
        if len(parts) >= 2:
            words.append(parts[-1])
    if len(words) != 7776:
        raise RuntimeError(f"EFF wordlist parse failed: expected 7776 words, got {len(words)}")
    return tuple(words)


def generate_passphrase(n_words: int = 6, separator: str = _DEFAULT_SEPARATOR) -> str:
    """Generate a Diceware passphrase from the bundled EFF wordlist.

    Each word is drawn independently with :func:`secrets.choice`, giving
    ``log2(7776) ≈ 12.92`` bits of entropy per word. Default 6 words
    yields ~77.5 bits — enough to make scrypt brute force infeasible.

    :param n_words: Number of words to draw. Must satisfy
        ``_MIN_WORDS <= n_words <= _MAX_WORDS`` (4–12 inclusive).
    :param separator: String joined between words. Default ``"-"``.
    :raises ValueError: If ``n_words`` is outside the allowed range or
        ``separator`` contains characters that would obscure word
        boundaries (newline, NUL).
    """
    if not _MIN_WORDS <= n_words <= _MAX_WORDS:
        raise ValueError(f"n_words must be between {_MIN_WORDS} and {_MAX_WORDS}, got {n_words}")
    if any(c in separator for c in ("\n", "\r", "\0")):
        raise ValueError("separator must not contain newline or NUL bytes")
    words = _load_wordlist()
    chosen = [secrets.choice(words) for _ in range(n_words)]
    return separator.join(chosen)


def estimate_entropy_bits(n_words: int) -> float:
    """Return the Shannon entropy in bits of an ``n_words`` passphrase
    drawn from the EFF large wordlist.

    Useful for "your passphrase has X bits" UI hints. Exact value is
    ``n_words * log2(7776)``; we round to one decimal for display.
    """
    import math

    if n_words <= 0:
        return 0.0
    return round(n_words * math.log2(7776), 1)
