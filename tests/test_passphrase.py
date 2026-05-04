"""Unit tests for ``axon.security.wordlist`` (PR E / v0.4.0 Item 1).

Coverage:
- wordlist parses to exactly 7776 entries
- ``generate_passphrase`` produces N words with correct separator
- ``generate_passphrase`` rejects out-of-range word counts and bad separators
- entropy estimate matches ``log2(7776) * n_words``
- ``secrets.choice`` yields high diversity (no duplicate phrases in 1000 runs at default 6 words)
- public re-exports on ``axon.security`` work
- REST ``/suggestions/passphrase`` returns the expected shape and rejects bad input
"""
from __future__ import annotations

import math

import pytest


def test_wordlist_loads_exactly_7776_words():
    from axon.security.wordlist import _load_wordlist

    words = _load_wordlist()
    assert len(words) == 7776
    # Spot-check first entry from EFF source list
    assert words[0] == "abacus"

    # All entries are non-empty; 4 EFF entries are hyphenated
    # (drop-down, felt-tip, t-shirt, yo-yo) so allow `-` in addition
    # to alphabetics.
    def _ok(w: str) -> bool:
        return bool(w) and all(c.isalpha() or c == "-" for c in w)

    assert all(_ok(w) for w in words)


def test_load_wordlist_is_cached():
    from axon.security.wordlist import _load_wordlist

    a = _load_wordlist()
    b = _load_wordlist()
    assert a is b  # tuple identity preserved by lru_cache


def test_generate_passphrase_default_six_words():
    from axon.security.wordlist import generate_passphrase

    p = generate_passphrase()
    parts = p.split(" ")
    assert len(parts) == 6
    # Each chosen word is non-empty (some EFF entries contain hyphens)
    assert all(part for part in parts)


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 10, 12])
def test_generate_passphrase_word_count(n):
    from axon.security.wordlist import generate_passphrase

    p = generate_passphrase(n_words=n)
    parts = p.split(" ")
    assert len(parts) == n


@pytest.mark.parametrize("bad", [0, 1, 2, 3, 13, 100, -1])
def test_generate_passphrase_rejects_out_of_range(bad):
    from axon.security.wordlist import generate_passphrase

    with pytest.raises(ValueError, match="n_words must be between 4 and 12"):
        generate_passphrase(n_words=bad)


def test_generate_passphrase_custom_separator():
    from axon.security.wordlist import generate_passphrase

    p = generate_passphrase(n_words=5, separator="-")
    # 4+ separators in result (some chosen words may be hyphenated)
    assert p.count("-") >= 4


def test_generate_passphrase_empty_separator_allowed():
    from axon.security.wordlist import generate_passphrase

    p = generate_passphrase(n_words=4, separator="")
    # All chars alpha or hyphen (hyphenated EFF entries possible)
    assert p
    assert all(c.isalpha() or c == "-" for c in p)


@pytest.mark.parametrize("bad_sep", ["\n", "a\nb", "x\0y", "\r"])
def test_generate_passphrase_rejects_bad_separator(bad_sep):
    from axon.security.wordlist import generate_passphrase

    with pytest.raises(ValueError, match="separator"):
        generate_passphrase(separator=bad_sep)


def test_entropy_estimate_matches_formula():
    from axon.security.wordlist import estimate_entropy_bits

    expected_per_word = math.log2(7776)
    for n in (1, 4, 6, 8, 12):
        got = estimate_entropy_bits(n)
        # Allow 0.1 bit drift from the rounding in estimate_entropy_bits
        assert abs(got - n * expected_per_word) < 0.1


def test_entropy_six_words_at_least_77_bits():
    """Plan target: ≥77 bits at 6 words."""
    from axon.security.wordlist import estimate_entropy_bits

    assert estimate_entropy_bits(6) >= 77.0


def test_entropy_zero_or_negative_returns_zero():
    from axon.security.wordlist import estimate_entropy_bits

    assert estimate_entropy_bits(0) == 0.0
    assert estimate_entropy_bits(-3) == 0.0


def test_no_duplicate_phrases_in_many_runs():
    """At 6 words there are 7776**6 ≈ 2.2e23 possible phrases; the
    chance of any collision in 1000 runs is astronomically small.
    A duplicate would mean ``secrets.choice`` is broken."""
    from axon.security.wordlist import generate_passphrase

    seen = {generate_passphrase() for _ in range(1000)}
    assert len(seen) == 1000


def test_security_module_reexports():
    """The public ``axon.security`` namespace exposes the helpers so
    downstream callers don't need to know about the ``wordlist``
    submodule."""
    from axon import security

    p = security.generate_passphrase(n_words=4)
    assert len(p.split(" ")) == 4
    assert security.estimate_entropy_bits(6) >= 77.0
    assert "generate_passphrase" in security.__all__
    assert "estimate_entropy_bits" in security.__all__


# ---------------------------------------------------------------------------
# REST surface
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from axon.api import app

    return TestClient(app, raise_server_exceptions=True)


def test_rest_suggest_passphrase_default(api_client):
    resp = api_client.get("/suggestions/passphrase")
    assert resp.status_code == 200
    body = resp.json()
    assert "passphrase" in body
    assert body["n_words"] == 6
    assert body["entropy_bits"] >= 77.0
    # API default mirrors the library default (space)
    assert body["separator"] == "-"  # endpoint kept "-" for URL-friendliness
    assert body["source"] == "eff_large_wordlist"


def test_rest_suggest_passphrase_custom_words(api_client):
    resp = api_client.get("/suggestions/passphrase?words=8")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_words"] == 8


def test_rest_suggest_passphrase_rejects_too_few_words(api_client):
    resp = api_client.get("/suggestions/passphrase?words=2")
    assert resp.status_code == 422


def test_rest_suggest_passphrase_rejects_too_many_words(api_client):
    resp = api_client.get("/suggestions/passphrase?words=50")
    assert resp.status_code == 422
