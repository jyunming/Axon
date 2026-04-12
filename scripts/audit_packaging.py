#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import tomllib  # py311+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def _load_toml(path: Path) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit packaging/version single-source configuration")
    ap.add_argument("--expected-version", default="", help="Optional version string to enforce")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    pyproject = _load_toml(root / "pyproject.toml")
    cargo = _load_toml(root / "src" / "axon" / "Cargo.toml")
    extension_package_json = root / "integrations" / "vscode-axon" / "package.json"
    website = root / "index.html"

    errors: list[str] = []

    build_backend = pyproject.get("build-system", {}).get("build-backend")
    if build_backend != "maturin":
        errors.append(f"build-backend must be 'maturin', got: {build_backend!r}")

    project = pyproject.get("project", {})
    if "version" in project:
        errors.append("project.version must not be set; version should be dynamic from Cargo.toml")

    dynamic = project.get("dynamic", [])
    if "version" not in dynamic:
        errors.append("project.dynamic must include 'version'")

    cargo_version = cargo.get("package", {}).get("version")
    if not cargo_version:
        errors.append("src/axon/Cargo.toml package.version is missing")

    try:
        ext_data = json.loads(extension_package_json.read_text(encoding="utf-8"))
        ext_version = ext_data.get("version")
    except Exception as exc:  # pragma: no cover
        ext_version = None
        errors.append(f"failed to read integrations/vscode-axon/package.json: {exc}")

    if ext_version != cargo_version:
        errors.append(
            "version mismatch: integrations/vscode-axon/package.json "
            f"has {ext_version!r}, Cargo.toml has {cargo_version!r}"
        )

    website_text = website.read_text(encoding="utf-8")
    hero_match = re.search(r"v(\d+\.\d+\.\d+)\s+—\s+now on PyPI", website_text)
    install_match = re.search(r"Successfully installed axon-rag-(\d+\.\d+\.\d+)", website_text)
    hero_version = hero_match.group(1) if hero_match else None
    install_version = install_match.group(1) if install_match else None

    if hero_version != cargo_version:
        errors.append(
            f"website version mismatch: index.html hero pill has {hero_version!r}, "
            f"Cargo.toml has {cargo_version!r}"
        )
    if install_version != cargo_version:
        errors.append(
            f"website version mismatch: index.html terminal install line has {install_version!r}, "
            f"Cargo.toml has {cargo_version!r}"
        )

    duplicate_manifest = root / "src" / "axon" / "axon_rust_Cargo.toml"
    if duplicate_manifest.exists():
        errors.append("duplicate manifest exists: src/axon/axon_rust_Cargo.toml")

    if args.expected_version and cargo_version != args.expected_version:
        errors.append(
            f"version mismatch: expected {args.expected_version!r}, Cargo.toml has {cargo_version!r}"
        )

    print(f"Cargo version: {cargo_version}")
    print(f"VS Code extension version: {ext_version}")
    print(f"Website hero version: {hero_version}")
    print(f"Website terminal version: {install_version}")
    if errors:
        print("Packaging audit FAILED:")
        for e in errors:
            print(f" - {e}")
        return 1

    print("Packaging audit PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
