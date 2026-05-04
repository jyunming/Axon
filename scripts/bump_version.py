#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SEMVER = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"  $ {' '.join(cmd)}  (cwd={cwd.name})")
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32"))


def _rebuild_and_bundle_vsix(root: Path, version: str) -> None:
    ext_dir = root / "integrations" / "vscode-axon"
    bundle_dir = root / "src" / "axon" / "extensions"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print("Rebuilding VS Code extension VSIX...")
    try:
        _run(["npm", "run", "package"], cwd=ext_dir)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  WARNING: VSIX rebuild failed ({exc}); bundled copy NOT refreshed.")
        print(f"  Run manually:   cd {ext_dir} && npm run package")
        return

    built = ext_dir / f"axon-copilot-{version}.vsix"
    if not built.exists():
        print(f"  WARNING: expected {built.name} not produced; skipping bundle refresh.")
        return

    for old in bundle_dir.glob("axon-copilot-*.vsix"):
        if old.name != built.name:
            print(f"  removing stale bundled VSIX: {old.name}")
            old.unlink()

    dest = bundle_dir / built.name
    shutil.copy2(built, dest)
    print(f"  bundled: {dest.relative_to(root)} ({dest.stat().st_size // 1024} KB)")


def _refresh_cargo_lock(root: Path) -> None:
    try:
        _run(
            ["cargo", "update", "-p", "axon", "--precise", _cargo_version(root)],
            cwd=root / "src" / "axon",
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # cargo may be absent in bump-only workflows; acceptable
        print("  note: cargo not available; Cargo.lock not refreshed (CI will update it).")


def _cargo_version(root: Path) -> str:
    cargo = root / "src" / "axon" / "Cargo.toml"
    for line in cargo.read_text(encoding="utf-8").splitlines():
        m = re.match(r'^version\s*=\s*"([^"]+)"', line)
        if m:
            return m.group(1)
    raise SystemExit("could not read Cargo version")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bump single-source package version in Cargo.toml")
    ap.add_argument("version", help="New version (e.g. 0.2.0)")
    ap.add_argument(
        "--skip-vsix", action="store_true", help="Skip VSIX rebuild (default: rebuild and bundle)"
    )
    ap.add_argument("--skip-cargo-lock", action="store_true", help="Skip Cargo.lock refresh")
    args = ap.parse_args()

    version = args.version.strip()
    if not SEMVER.match(version):
        raise SystemExit(f"Invalid version format: {version}")

    root = Path(__file__).resolve().parents[1]
    cargo = root / "src" / "axon" / "Cargo.toml"
    text = cargo.read_text(encoding="utf-8")

    new_text, n = re.subn(
        r'^version\s*=\s*"[^"]+"\s*$', f'version = "{version}"', text, count=1, flags=re.M
    )
    if n != 1:
        raise SystemExit("Failed to update version in src/axon/Cargo.toml")

    cargo.write_text(new_text, encoding="utf-8")

    extension_package_json = root / "integrations" / "vscode-axon" / "package.json"
    ext_data = json.loads(extension_package_json.read_text(encoding="utf-8"))
    ext_data["version"] = version
    extension_package_json.write_text(json.dumps(ext_data, indent=2) + "\n", encoding="utf-8")

    # v0.4.0 PR I: index.html is now rendered from index.template.html
    # via scripts/render_index.py — every release-version slot in the
    # landing page funnels through the {{AXON_VERSION}} placeholder.
    # We invoke the renderer as a subprocess to keep this script's
    # import surface zero (the renderer is standalone-importable).
    render_script = root / "scripts" / "render_index.py"
    if render_script.exists():
        subprocess.run(
            [sys.executable, str(render_script), "--version", version],
            check=True,
            cwd=root,
        )
    else:
        print(
            f"  WARNING: {render_script.relative_to(root)} missing; "
            "index.html version slots NOT refreshed."
        )

    print(f"Updated version to {version} in:")
    print(" - src/axon/Cargo.toml")
    print(" - integrations/vscode-axon/package.json")
    print(" - index.html (via render_index.py)")

    if not args.skip_vsix:
        _rebuild_and_bundle_vsix(root, version)
    if not args.skip_cargo_lock:
        _refresh_cargo_lock(root)

    print(f"\nNext: run `python scripts/audit_packaging.py --expected-version {version}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
