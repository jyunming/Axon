#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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

    extension_package_json = root / "integrations" / "vscode-axon" / "package.json"
    ext_data = json.loads(extension_package_json.read_text(encoding="utf-8"))
    ext_data["version"] = version
    extension_package_json.write_text(json.dumps(ext_data, indent=2) + "\n", encoding="utf-8")

    website = root / "index.html"
    website_text = website.read_text(encoding="utf-8")
    website_text, n1 = re.subn(
        r"(v)\d+\.\d+\.\d+(\s+—\s+now on PyPI)",
        rf"\g<1>{version}\g<2>",
        website_text,
        count=1,
    )
    website_text, n2 = re.subn(
        r"(Successfully installed axon-rag-)\d+\.\d+\.\d+",
        rf"\g<1>{version}",
        website_text,
        count=1,
    )
    if n1 != 1 or n2 != 1:
        raise SystemExit("Failed to update version strings in index.html")
    website.write_text(website_text, encoding="utf-8")

    print(f"Updated version to {version} in:")
    print(" - src/axon/Cargo.toml")
    print(" - integrations/vscode-axon/package.json")
    print(" - index.html")
    print("Next: run `python scripts/audit_packaging.py --expected-version <version>`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
