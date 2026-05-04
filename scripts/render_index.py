#!/usr/bin/env python
"""Render ``index.html`` from ``index.template.html`` + the canonical
package version.

v0.4.0 PR I — single-source-of-truth for the landing page version.

Before this script existed, every release bump had to chase 5+
hand-edited version strings in ``index.html``. The Copilot review on
PR #104 caught two stragglers; the user (correctly) flagged this as a
maintenance burden. Now the workflow is:

1. Edit ``index.template.html`` — substitute every CURRENT-release
   version string with ``{{AXON_VERSION}}``. Historical references
   (e.g. "v0.3.2 graph backend changes" educational content) stay
   verbatim — they're attribution, not staleness.
2. Bump ``src/axon/Cargo.toml`` (or run ``scripts/bump_version.py X.Y.Z``).
3. Run ``python scripts/render_index.py`` (or let bump_version.py
   call it for you).
4. ``scripts/audit_packaging.py`` calls ``--check`` to fail if the
   committed ``index.html`` drifted from the rendered template.

Why a checked-in rendered file (not template + build-time render)?
GitHub Pages serves the repo root directly with no build step. We
keep the rendered file committed so the Pages site always works,
and the audit catches drift. CI can run ``--check`` to enforce.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PLACEHOLDER = "{{AXON_VERSION}}"


def _read_cargo_version(root: Path) -> str:
    """Return the canonical version from ``src/axon/Cargo.toml``.

    We avoid pulling in ``tomllib`` for this one-line read because
    bump scripts run before maturin builds finish — keeping this
    importable on minimal Python is worth a regex.
    """
    cargo = root / "src" / "axon" / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    if not m:
        raise SystemExit(f"Could not find version in {cargo}")
    return m.group(1)


def _render(template_path: Path, version: str) -> str:
    """Substitute every ``{{AXON_VERSION}}`` in *template_path* with *version*."""
    text = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in text:
        raise SystemExit(
            f"Template {template_path} has no {PLACEHOLDER} placeholder — "
            "did the file get accidentally rendered into the template slot?"
        )
    return text.replace(PLACEHOLDER, version)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Render index.html from index.template.html and Cargo.toml version"
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="Compare rendered output to index.html on disk; exit 1 on drift "
        "(no write). Used by audit_packaging and CI to catch stale renders.",
    )
    ap.add_argument(
        "--version",
        default="",
        help="Override the version string (default: read from src/axon/Cargo.toml)",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    template = root / "index.template.html"
    output = root / "index.html"

    if not template.exists():
        raise SystemExit(f"Missing template: {template}")

    version = args.version.strip() or _read_cargo_version(root)
    rendered = _render(template, version)

    if args.check:
        on_disk = output.read_text(encoding="utf-8") if output.exists() else ""
        if on_disk == rendered:
            print(f"index.html is in sync with template (version={version})")
            return 0
        print(
            f"index.html is OUT OF SYNC with template (expected version={version}).\n"
            "Run: python scripts/render_index.py"
        )
        return 1

    output.write_text(rendered, encoding="utf-8")
    placeholders_replaced = template.read_text(encoding="utf-8").count(PLACEHOLDER)
    print(
        f"Rendered {output.relative_to(root)} from "
        f"{template.relative_to(root)} (version={version}, "
        f"{placeholders_replaced} substitution(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
