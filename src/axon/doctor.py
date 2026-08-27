"""``axon doctor`` — first-run health diagnostics.

A non-destructive sanity check that surfaces the most common
"Axon-doesn't-work-yet" failure modes up front, instead of letting a
new user discover them on their first query:

    * Python version too old for the project floor.
    * Ollama daemon not running on the configured host/port.
    * Default LLM model not pulled to the local Ollama cache.
    * Configured embedding model not present locally.
    * Store / project base directory not writable.
    * Optional extras absent (sealed, ui, loaders) — informational only.

The doctor returns one of three exit shapes:

    "ok"       — every required check passed.
    "warning"  — every required check passed but something optional
                 is degraded (e.g. recommended extra missing).
    "error"    — at least one required check failed; the user must act
                 before Axon will work end-to-end.

Each check returns a :class:`Check` whose ``status`` is ``"ok"`` /
``"warning"`` / ``"error"`` and whose ``hint`` (when non-empty) is the
single most actionable next step. The CLI renders these as a colored
checklist, but the dataclass is also importable so other surfaces
(REPL ``/doctor``, the setup wizard, future GUIs) can reuse the logic.
"""
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Minimum Python version Axon supports. Mirrors ``python_requires`` in
# pyproject.toml; bumped together at release time.
_MIN_PYTHON = (3, 10)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class Check:
    """Single doctor check result."""

    name: str
    status: str  # "ok" | "warning" | "error"
    detail: str = ""
    hint: str = ""

    @property
    def passed(self) -> bool:
        return self.status == "ok"


@dataclass
class DoctorReport:
    """Aggregate report across every doctor check."""

    overall: str  # "ok" | "warning" | "error"
    checks: list[Check] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Individual checks (each is independent; any may be skipped without
# breaking the others, e.g. when Ollama is intentionally not used).
# ---------------------------------------------------------------------------


def check_python_version() -> Check:
    cur = sys.version_info[:2]
    if cur >= _MIN_PYTHON:
        return Check(
            "Python version",
            "ok",
            detail=f"{sys.version.split()[0]} ≥ {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}",
        )
    return Check(
        "Python version",
        "error",
        detail=f"{sys.version.split()[0]} < {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}",
        hint=(
            f"Upgrade to Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ "
            "(see https://www.python.org/downloads/)."
        ),
    )


def check_ollama_reachable(base_url: str | None = None) -> Check:
    """Probe the Ollama daemon at the configured base URL.

    Returns ``ok`` when the HTTP root responds, ``warning`` when the
    request fails (Ollama may simply not be in use — Axon supports
    OpenAI / vLLM / Gemini / Grok cloud providers too).
    """
    url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    try:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            ok = 200 <= resp.status < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        ok = False
    if ok:
        return Check("Ollama reachable", "ok", detail=url)
    return Check(
        "Ollama reachable",
        "warning",
        detail=f"no response from {url}",
        hint=(
            "Start the Ollama app, run `ollama serve`, or set OLLAMA_HOST. "
            "Skip if you only use cloud providers (OpenAI / Gemini / Grok)."
        ),
    )


def check_default_model(
    model_name: str | None = None,
    base_url: str | None = None,
) -> Check:
    """Verify the LLM model is in the local Ollama cache."""
    model = model_name or os.environ.get("AXON_DEFAULT_MODEL") or "llama3.1:8b"
    url = (base_url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
    tags_url = f"{url}/api/tags"
    try:
        import json
        import urllib.error
        import urllib.request

        with urllib.request.urlopen(tags_url, timeout=2.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        names = {entry.get("name", "") for entry in payload.get("models", [])}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return Check(
            "LLM model pulled",
            "warning",
            detail=f"could not query {tags_url}",
            hint="Resolve the Ollama-reachable check first.",
        )
    if model in names:
        return Check("LLM model pulled", "ok", detail=model)
    return Check(
        "LLM model pulled",
        "warning",
        detail=f"{model} not found in local cache",
        hint=f"Run `ollama pull {model}` (~4–8 GB download).",
    )


def check_store_writable(store_base: str | None = None) -> Check:
    """Ensure the AxonStore base directory exists and is writable."""
    base = (
        store_base or os.environ.get("AXON_STORE_BASE") or str(Path.home() / ".axon" / "AxonStore")
    )
    expanded = os.path.expanduser(base)
    try:
        Path(expanded).mkdir(parents=True, exist_ok=True)
        # Atomic write probe — never leaves files behind on success.
        probe = Path(expanded) / ".doctor_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(
            "Store path writable",
            "error",
            detail=f"{expanded}: {exc}",
            hint=(
                "Create the directory yourself or set AXON_STORE_BASE to a "
                "writable location (e.g. an external drive)."
            ),
        )
    return Check("Store path writable", "ok", detail=expanded)


def check_optional_extras() -> Check:
    """Inform whether the most user-visible extras are installed.

    Always non-fatal — Axon works without them; the doctor just nudges
    new users toward ``[starter]`` so they don't hit ImportErrors later.

    ``streamlit`` is deliberately NOT checked here: the Streamlit UI is
    deprecated and was dropped from ``[starter]``, so warning about it would
    point users at a bundle that no longer ships it. The maintained browser
    surface is the web GUI served by ``axon-api`` at ``/gui/``, which needs no
    extra dependency.
    """
    missing: list[str] = []
    try:
        import cryptography  # noqa: F401
        import keyring  # noqa: F401
    except ImportError:
        missing.append("cryptography + keyring (sealed sharing)")
    if not missing:
        return Check(
            "Recommended extras",
            "ok",
            detail="cryptography, keyring all available",
        )
    return Check(
        "Recommended extras",
        "warning",
        detail="missing: " + ", ".join(missing),
        hint="Install the starter bundle: pip install 'axon-rag[starter]'",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def check_local_llm_reachable(
    provider: str | None, base_url: str | None, api_key: str | None = None
) -> Check:
    """Ping the configured local OpenAI-compatible endpoint and list its models.

    Only meaningful when ``llm_provider == "local"``; skipped (as "ok") otherwise
    so the doctor stays quiet for people not using it.

    ``api_key`` mirrors ``llm.local_api_key``. Most local servers ignore auth, but
    a gated one (LM Studio with a key set, a reverse proxy) would 401 an
    unauthenticated probe and the doctor would report "unreachable" for an
    endpoint that is actually fine — so send the same header the real client does.

    Axon never loads models itself, so "reachable but serving nothing" is a real
    and common state — llama-server in router mode answers ``/models`` before any
    model is resident. That earns a warning, not an error: the endpoint is fine,
    but the next query would fail.
    """
    if provider != "local":
        return Check(
            "Local LLM endpoint",
            "ok",
            detail="not in use (llm_provider is not 'local')",
        )
    url = (base_url or "").rstrip("/")
    if not url:
        return Check(
            "Local LLM endpoint",
            "error",
            detail="llm_provider is 'local' but local_base_url is empty",
            hint="Set llm.local_base_url in config.yaml, or AXON_LOCAL_LLM_BASE_URL.",
        )
    try:
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = httpx.get(f"{url}/models", timeout=5.0, headers=headers)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001 - the doctor reports, never raises
        return Check(
            "Local LLM endpoint",
            "error",
            detail=f"{url} unreachable ({type(exc).__name__})",
            hint=(
                "Start your local LLM server (e.g. `las start` for llama.cpp, or "
                "`vllm serve <model>`), then re-run. Axon does not start it for you."
            ),
        )
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    models: list[str] = [
        str(r["id"]) for r in rows if isinstance(r, dict) and r.get("id") is not None
    ]
    if not models:
        return Check(
            "Local LLM endpoint",
            "warning",
            detail=f"{url} is up but reports no models",
            hint="Load a model on the server — Axon will not load one for you.",
        )
    shown = ", ".join(models[:3]) + (f" (+{len(models) - 3} more)" if len(models) > 3 else "")
    return Check("Local LLM endpoint", "ok", detail=f"{url} — {shown}")


def check_update_available(offline: bool = False) -> Check:
    """Query PyPI (rate-limited, cached) for a newer axon-rag release.

    Always non-fatal — an update being available, or the check itself
    failing, is informational only. Reports ``ok`` (not ``warning``) when
    offline_mode is on or the check fails, matching this check's other
    network-dependent siblings' behavior of staying non-alarming rather
    than nagging about something the user has explicitly opted out of —
    the ``detail`` field still says why, just at ``ok`` severity.
    """
    from axon.update_check import check_for_update

    result = check_for_update(offline=offline)
    if result.skipped_reason == "offline_mode":
        return Check("Update available", "ok", detail="skipped (offline_mode)")
    if result.skipped_reason == "error":
        return Check("Update available", "ok", detail="couldn't reach PyPI")
    if result.update_available:
        return Check(
            "Update available",
            "warning",
            detail=f"{result.current} → {result.latest}",
            hint="Run `axon update` to upgrade the package and VS Code extension together.",
        )
    return Check("Update available", "ok", detail=f"{result.current} is current")


_CHECK_FUNCS: tuple[Callable[..., Check], ...] = (
    check_python_version,
    check_ollama_reachable,
    check_default_model,
    check_store_writable,
    check_local_llm_reachable,
    check_optional_extras,
    check_update_available,
)


def run_doctor(config: Any | None = None) -> DoctorReport:
    """Run every doctor check and return a single :class:`DoctorReport`.

    ``config`` is an optional :class:`AxonConfig` — when provided, the
    Ollama base URL, default model, and store base are pulled from it
    rather than from environment variables, so the doctor checks the
    same values the running brain would use.
    """
    base_url: str | None = None
    model_name: str | None = None
    store_base: str | None = None
    llm_provider: str | None = None
    local_base_url: str | None = None
    local_api_key: str | None = None
    if config is not None:
        # AxonConfig is a flat dataclass: llm_provider / llm_model / ollama_base_url
        # / axon_store_base. We also tolerate nested-attribute fixtures (e.g.
        # SimpleNamespace(llm=..., store=...) in tests) and an older-style
        # default_model / store_base fallback for back-compat.
        base_url = getattr(config, "ollama_base_url", None) or getattr(
            getattr(config, "llm", None), "base_url", None
        )
        model_name = (
            getattr(config, "llm_model", None)
            or getattr(getattr(config, "llm", None), "model", None)
            or getattr(config, "default_model", None)
        )
        store_base = (
            getattr(config, "axon_store_base", None)
            or getattr(getattr(config, "store", None), "base", None)
            or getattr(config, "store_base", None)
        )
        llm_provider = getattr(config, "llm_provider", None) or getattr(
            getattr(config, "llm", None), "provider", None
        )
        local_base_url = getattr(config, "local_base_url", None) or getattr(
            getattr(config, "llm", None), "local_base_url", None
        )
        local_api_key = getattr(config, "local_api_key", None) or getattr(
            getattr(config, "llm", None), "local_api_key", None
        )
    offline_mode = bool(getattr(config, "offline_mode", False)) if config is not None else False

    checks: list[Check] = []
    for fn in _CHECK_FUNCS:
        if fn is check_ollama_reachable:
            checks.append(fn(base_url))
        elif fn is check_default_model:
            checks.append(fn(model_name, base_url))
        elif fn is check_store_writable:
            checks.append(fn(store_base))
        elif fn is check_local_llm_reachable:
            checks.append(fn(llm_provider, local_base_url, local_api_key))
        elif fn is check_update_available:
            checks.append(fn(offline_mode))
        else:
            checks.append(fn())
    if any(c.status == "error" for c in checks):
        overall = "error"
    elif any(c.status == "warning" for c in checks):
        overall = "warning"
    else:
        overall = "ok"
    return DoctorReport(overall=overall, checks=checks)


def render_report(report: DoctorReport, *, use_color: bool | None = None) -> str:
    """Render the report as a multiline checklist for terminal output."""
    if use_color is None:
        use_color = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    icon = {"ok": "✓", "warning": "!", "error": "✗"}
    color = {
        "ok": "\033[32m",
        "warning": "\033[33m",
        "error": "\033[31m",
    }
    reset = "\033[0m"
    width = shutil.get_terminal_size((80, 20)).columns
    lines = ["", "  Axon doctor"]
    for check in report.checks:
        marker = icon[check.status]
        prefix = f"  {color[check.status]}{marker}{reset} " if use_color else f"  {marker} "
        line = f"{prefix}{check.name}"
        if check.detail:
            line += f"  —  {check.detail}"
        lines.append(line[: max(width - 1, 40)])
        if check.hint and check.status != "ok":
            lines.append(f"        {check.hint}")
    summary_color = color[report.overall] if use_color else ""
    summary = {
        "ok": "All checks passed.",
        "warning": "Some optional checks need attention.",
        "error": "At least one required check failed.",
    }[report.overall]
    lines.append("")
    lines.append(f"  {summary_color}{summary}{reset if use_color else ''}")
    return "\n".join(lines)
