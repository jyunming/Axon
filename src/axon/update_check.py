"""axon update — PyPI version check plus an optional automated upgrade.

Two distinct operations live here, deliberately kept separate:

* ``check_for_update()`` — read-only. Queries PyPI, compares against the
  installed version, and (by default) rate-limits itself via an on-disk
  cache. Never touches the environment. Shared by the startup hooks in
  ``cli.py``/``repl.py``/``api.py``, ``axon --doctor``, and ``axon update``
  itself (which always bypasses the cache for a live answer).
* ``run_update()`` — the one function in this module allowed to mutate the
  environment. Detects the install method (pip / pipx / conda), runs the
  appropriate upgrade command, then re-installs the VS Code extension
  (already bundled inside the freshly-upgraded package — see
  ``axon.ext_install``) so both artifacts land in sync. Refuses outright
  inside Docker or when an ``axon-api`` server is currently live against
  the active store; the caller (``cli.py``) is responsible for the
  confirm-before-executing gate.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PYPI_URL = "https://pypi.org/pypi/axon-rag/json"
_CHECK_TIMEOUT_S = 5.0
_CACHE_TTL_S = 24 * 60 * 60  # 1 day
_CACHE_PATH = Path.home() / ".axon" / ".update_check_cache.json"

PACKAGE_NAME = "axon-rag"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def current_version() -> str:
    """Return the installed axon-rag version, or a dev placeholder."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.0.0+dev"


def _parse_numeric_prefix(v: str) -> tuple[int, ...] | None:
    """Parse a leading dotted-integer run (e.g. '0.4.4' from '0.4.4rc1').

    Deliberately simple rather than a full PEP 440 parser — this project's
    releases are plain X.Y.Z (see CLAUDE.md's Versions section). Returns
    None when the string doesn't start with at least one integer component,
    so callers can fall back to a string-inequality comparison.
    """
    m = re.match(r"^(\d+(?:\.\d+)*)", v.strip())
    if not m:
        return None
    return tuple(int(part) for part in m.group(1).split("."))


def is_newer(latest: str, current: str) -> bool:
    """Return True if *latest* is a newer version than *current*."""
    lp, cp = _parse_numeric_prefix(latest), _parse_numeric_prefix(current)
    if lp is not None and cp is not None:
        if lp != cp:
            return lp > cp
        # Same numeric prefix (e.g. both "0.4.4") — a longer/differently
        # suffixed string (e.g. "0.4.4" vs "0.4.4rc1") is treated as equal
        # rather than guessing pre-release ordering; not worth a full PEP
        # 440 parser for this project's plain-semver release history.
        return False
    return latest != current and latest > current


# ---------------------------------------------------------------------------
# Rate-limit cache
# ---------------------------------------------------------------------------


def _read_cache() -> dict | None:
    try:
        data = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_cache(latest_version: str) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(
            json.dumps({"checked_at": time.time(), "latest_version": latest_version}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _cache_is_fresh(cache: dict, ttl_s: float) -> bool:
    checked_at = cache.get("checked_at")
    if not isinstance(checked_at, int | float):
        return False
    return (time.time() - checked_at) < ttl_s


# ---------------------------------------------------------------------------
# The check itself
# ---------------------------------------------------------------------------


@dataclass
class UpdateCheckResult:
    current: str
    latest: str | None  # None means the check failed or was skipped
    update_available: bool
    skipped_reason: str | None = None  # "offline_mode" | "error" | None
    from_cache: bool = False


def check_for_update(
    *,
    offline: bool = False,
    force: bool = False,
    timeout: float = _CHECK_TIMEOUT_S,
    ttl_s: float = _CACHE_TTL_S,
) -> UpdateCheckResult:
    """Return whether a newer axon-rag release exists on PyPI.

    Never raises — network/parse failures come back as a result with
    ``skipped_reason="error"`` so callers can fail quiet, not fail loud.
    ``offline=True`` (config.offline_mode) short-circuits before any
    network access. ``force=True`` (used by ``axon update``) bypasses the
    on-disk rate-limit cache for a live answer.
    """
    cur = current_version()
    if offline:
        return UpdateCheckResult(cur, None, False, skipped_reason="offline_mode")

    if not force:
        cache = _read_cache()
        if cache and _cache_is_fresh(cache, ttl_s):
            latest = cache.get("latest_version")
            if isinstance(latest, str):
                return UpdateCheckResult(cur, latest, is_newer(latest, cur), from_cache=True)

    try:
        import httpx

        resp = httpx.get(_PYPI_URL, timeout=timeout)
        resp.raise_for_status()
        latest = resp.json()["info"]["version"]
    except Exception:
        return UpdateCheckResult(cur, None, False, skipped_reason="error")

    _write_cache(latest)
    return UpdateCheckResult(cur, latest, is_newer(latest, cur))


def format_suggestion(result: UpdateCheckResult) -> str | None:
    """One-line suggestion for startup banners, or None when there's nothing
    to say (up to date, check skipped/failed — silence is intentional for
    the passive surfaces, see module docstring)."""
    if not result.update_available or not result.latest:
        return None
    return f"  Update available: {result.current} → {result.latest}  ·  run `axon update`"


# ---------------------------------------------------------------------------
# Install-method detection
# ---------------------------------------------------------------------------


def detect_install_method() -> str:
    """Best-effort guess at how axon-rag was installed: 'pipx', 'conda', or
    'pip' (the fallback when neither of the other two is detected)."""
    if os.environ.get("PIPX_HOME") or "pipx" in str(Path(sys.prefix)).lower():
        return "pipx"
    if os.environ.get("CONDA_PREFIX") or os.environ.get("CONDA_DEFAULT_ENV"):
        return "conda"
    return "pip"


def is_running_in_docker() -> bool:
    """Best-effort container detection — see run_update()'s Docker bail-out."""
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup = Path("/proc/1/cgroup")
        if cgroup.exists() and "docker" in cgroup.read_text(encoding="utf-8", errors="ignore"):
            return True
    except OSError:
        pass
    return False


def upgrade_command_for(method: str) -> list[str]:
    """The subprocess argv that performs the upgrade for *method*."""
    if method == "pipx":
        return ["pipx", "upgrade", PACKAGE_NAME]
    if method == "conda":
        return ["conda", "update", "-y", PACKAGE_NAME]
    return ["pip", "install", "-U", PACKAGE_NAME]


# ---------------------------------------------------------------------------
# axon update — the one function here that mutates the environment
# ---------------------------------------------------------------------------


@dataclass
class UpdateRunResult:
    status: str  # "already_current" | "refused" | "upgraded" | "failed"
    detail: str
    package_before: str = ""
    package_after: str = ""
    vsix_status: str = ""  # "" when not attempted


def run_update(
    *,
    config: Any = None,
    force_check: bool = True,
) -> UpdateRunResult:
    """Perform the full ``axon update`` sequence. Caller (cli.py) owns the
    confirmation prompt/-y gate — this function assumes it's already been
    granted and just executes.
    """
    offline = bool(getattr(config, "offline_mode", False))
    check = check_for_update(offline=offline, force=force_check)
    if check.skipped_reason == "offline_mode":
        return UpdateRunResult("refused", "offline_mode is on — skipping the network check.")
    if check.skipped_reason == "error" or not check.latest:
        return UpdateRunResult("failed", "Could not reach PyPI to check the latest version.")
    if not check.update_available:
        return UpdateRunResult(
            "already_current",
            f"Already on the latest version ({check.current}).",
            package_before=check.current,
            package_after=check.current,
        )

    if is_running_in_docker():
        return UpdateRunResult(
            "refused",
            "Running inside a container — `pip install` here would upgrade a "
            "filesystem layer that reverts on the next restart. Use "
            "`docker compose pull && docker compose up -d` instead.",
        )

    if config is not None:
        from axon import server_client as _sc

        live = _sc.find_live_server_for_store(config)
        if live:
            return UpdateRunResult(
                "refused",
                f"An axon-api server is currently live for this store "
                f"({live.get('host')}:{live.get('port')}, pid {live.get('pid')}). "
                "Stop it first — upgrading its package out from under a running "
                "server risks corrupting in-flight requests.",
            )

    import subprocess

    method = detect_install_method()
    argv = upgrade_command_for(method)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return UpdateRunResult("failed", f"Upgrade command failed to run: {exc}")
    if result.returncode != 0:
        return UpdateRunResult(
            "failed",
            f"`{' '.join(argv)}` exited {result.returncode}:\n{result.stderr.strip()}",
        )

    package_after = current_version()
    vsix_status = _reinstall_vscode_extension()
    return UpdateRunResult(
        "upgraded",
        f"Upgraded {check.current} → {package_after} via {method}.",
        package_before=check.current,
        package_after=package_after,
        vsix_status=vsix_status,
    )


def _reinstall_vscode_extension() -> str:
    """Re-run the bundled VSIX installer in-process (not a subprocess
    shell-out to `axon-ext`) so it reads the just-upgraded package's
    bundled VSIX. Failure here is a partial-success, not a fatal one —
    ext_install.py already degrades gracefully when `code` isn't on PATH.
    """
    try:
        from axon.ext_install import _find_code_cmd, _find_vsix
    except Exception as exc:
        return f"skipped (could not import installer: {exc})"
    try:
        vsix_path = _find_vsix()
    except FileNotFoundError as exc:
        return f"skipped ({exc})"
    code_cmd = _find_code_cmd()
    if not code_cmd:
        return f"skipped ('code' CLI not found — install manually: code --install-extension {vsix_path})"
    import subprocess

    result = subprocess.run(
        [code_cmd, "--install-extension", str(vsix_path)], capture_output=True, text=True
    )
    if result.returncode == 0:
        return "installed"
    return f"failed ({result.stderr.strip()[:200]})"
