#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bump single-source package version in Cargo.toml")
    ap.add_argument("version", help="New version (e.g. 0.2.0)")
    args = ap.parse_args()

    version = args.version.strip()
    if not SEMVER.match(version):
        raise SystemExit(f"Invalid version format: {version}")

    root = Path(__file__).resolve().parents[1]
    cargo = root / "src" / "axon" / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")

    new_text, n = re.subn(r'^version\s*=\s*"[^"]+"\s*$', f'version = "{version}"', text, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("Failed to update version in src/axon/Cargo.toml")

    cargo.write_text(new_text, encoding="utf-8")
    print(f"Updated Cargo version to {version}")
    print("Next: run `python scripts/audit_packaging.py --expected-version <version>`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
