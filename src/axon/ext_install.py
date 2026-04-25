"""
axon-ext — install the bundled Axon Copilot VS Code extension.

Usage:
    axon-ext                  # install
    axon-ext --path           # print VSIX path and exit
    axon-ext --version        # print extension version and exit
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.resources import files
from pathlib import Path


def _find_vsix() -> Path:
    """Return the path to the bundled VSIX file."""
    ext_dir = files("axon.extensions")
    vsix_files = [f for f in ext_dir.iterdir() if str(f).endswith(".vsix")]
    if not vsix_files:
        raise FileNotFoundError(
            "No VSIX file found in axon.extensions. "
            "The package may be incomplete — try reinstalling: pip install --force-reinstall axon-rag"
        )
    # If multiple versions exist (shouldn't happen), pick the latest by name.
    return Path(str(sorted(vsix_files, key=str)[-1]))


def _find_code_cmd() -> str | None:
    for candidate in ("code", "code-insiders"):
        if shutil.which(candidate):
            return candidate
    return None


def install_vscode_extension() -> None:
    """Entry point for the `axon-ext` CLI command."""
    args = sys.argv[1:]
    try:
        vsix_path = _find_vsix()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    if "--path" in args:
        print(vsix_path)
        return
    if "--version" in args:
        print(vsix_path.stem.removeprefix("axon-copilot-"))
        return
    code_cmd = _find_code_cmd()
    if not code_cmd:
        print(
            "ERROR: 'code' command not found.\n"
            "Make sure VS Code is installed and 'code' is on your PATH.\n"
            f"\nYou can also install manually:\n  code --install-extension {vsix_path}",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Installing Axon Copilot extension from {vsix_path} ...")
    result = subprocess.run(
        [code_cmd, "--install-extension", str(vsix_path)],
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode == 0:
        print("Done. Restart VS Code if it is currently open.")
    else:
        print(f"ERROR: Installation failed.\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
