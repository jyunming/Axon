"""``axon --project-pack`` / ``axon --project-unpack`` — zip a project's
entire on-disk footprint for backup, restore, or relocation.

Genuinely new functionality: no export/import/backup mechanism existed
anywhere in this codebase before this module. Patterned on
``axon.security.seal``'s cross-surface wiring style — the core functions
here are brain-agnostic (project name / zip path + an explicit
``user_dir`` in, no ``AxonBrain`` needed) so CLI's early-exit flag
handlers, which run before ``AxonBrain`` is constructed, can call these
directly.

Sealed projects: pack copies ``.security/`` verbatim (ciphertext + DEK
wraps); unpack restores it still sealed. The DEK wrap is bound to the
store's passphrase-derived master key, not the machine — a same-store
restore round-trips correctly with the original passphrase via the
normal ``security_unlock`` flow afterward. This is not a claim of
arbitrary cross-store/cross-owner portability.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import stat
import time
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from .projects import _parse_name

__all__ = ["ProjectPackError", "pack_project", "unpack_project"]

MANIFEST_NAME = "_axon_pack_manifest.json"
PACK_FORMAT_VERSION = 1
_SEALED_MARKER_REL = ".security/.sealed"
_SKELETON_DIRS = ("vector_store_data", "bm25_index", "sessions")


class ProjectPackError(Exception):
    """Raised for any pack/unpack failure — bad name, missing project,
    unsafe archive contents, or an existing unpack target without force."""


def _resolve_project_dir(name: str, user_dir: Path) -> Path:
    """Mirror axon.projects.project_dir()'s subs/ nesting, anchored at an
    explicit user_dir. Duplicated (not imported) for the same reason
    axon.security.seal._resolve_project_dir is: callers on the CLI
    early-exit path never call set_projects_root(), so the process-global
    axon.projects.PROJECTS_ROOT may not reflect the configured store.
    """
    segments = _parse_name(name)
    project_dir = Path(user_dir) / segments[0]
    for seg in segments[1:]:
        project_dir = project_dir / "subs" / seg
    return project_dir


def _is_project_sealed(project_dir: Path) -> bool:
    """Cheap filesystem probe — deliberately NOT axon.security.seal's
    version, which pulls in cryptography/keyring at import time. Keeps
    pack/unpack usable without the 'sealed' extra installed."""
    return (project_dir / _SEALED_MARKER_REL).is_file()


def _graphs_root() -> Path:
    """~/.axon/graphs — the external home for a relocated .dynamic_graph.db.
    A function (not a module constant) so tests can monkeypatch it."""
    return Path.home() / ".axon" / "graphs"


def _external_dynamic_graph_db(project_dir: Path) -> Path | None:
    """Return the live .dynamic_graph.db path if it's been relocated
    outside the project dir (graph_backends/dynamic_graph_backend.py's
    _resolve_db_path, for cloud-sync/network/WSL-mount bm25_index paths),
    else None. Mirrors that function's two-tier project_id resolution so
    pack finds the DB even for a sealed project (meta.json unreadable).
    """
    base = project_dir / "bm25_index"
    candidates: list[Path] = []
    try:
        meta = json.loads((project_dir / "meta.json").read_text(encoding="utf-8"))
        pid = (meta.get("project_id") or "").strip()
        if pid:
            candidates.append(_graphs_root() / pid / ".dynamic_graph.db")
    except Exception:
        pass
    resolved = str(base.resolve()) if base.exists() else str(base)
    fallback_id = hashlib.sha256(resolved.encode()).hexdigest()[:16]
    candidates.append(_graphs_root() / fallback_id / ".dynamic_graph.db")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def pack_project(
    project_name: str,
    user_dir: Path,
    *,
    out_path: Path | str | None = None,
) -> dict[str, Any]:
    """Zip *project_name*'s entire on-disk footprint.

    Brain-agnostic: the caller owns switching away from *project_name*
    first if it's the currently active project — a live writer mid-pack
    would produce a torn copy. This function only touches the filesystem.

    Returns ``{"status": "packed", "project", "out_path", "sealed",
    "file_count", "bytes"}``.
    """
    try:
        _parse_name(project_name)
    except ValueError as exc:
        raise ProjectPackError(str(exc)) from exc
    project_dir = _resolve_project_dir(project_name, Path(user_dir))
    if not project_dir.is_dir():
        raise ProjectPackError(f"Project does not exist: {project_name}")
    sealed = _is_project_sealed(project_dir)

    if out_path is None:
        safe_name = project_name.replace("/", "__")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]
        out_path = Path.home() / ".axon" / "packs" / f"{safe_name}-{stamp}.axonpack.zip"
    else:
        out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    external_db = _external_dynamic_graph_db(project_dir)
    local_db_rel = Path("bm25_index") / ".dynamic_graph.db"

    tmp_path = out_path.parent / f".{out_path.name}.packing"
    file_count = 0
    total_bytes = 0
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                "pack_format_version": PACK_FORMAT_VERSION,
                "axon_rag_version": _pkg_version("axon-rag"),
                "original_project": project_name,
                "packed_at": datetime.now(timezone.utc).isoformat(),
                "sealed": sealed,
            }
            zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
            for src in sorted(project_dir.rglob("*")):
                if not src.is_file():
                    continue
                rel = src.relative_to(project_dir)
                if external_db is not None and rel == local_db_rel:
                    continue  # superseded by the external, authoritative copy below
                zf.write(src, rel.as_posix())
                file_count += 1
                total_bytes += src.stat().st_size
            if external_db is not None:
                zf.write(external_db, local_db_rel.as_posix())
                file_count += 1
                total_bytes += external_db.stat().st_size
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    return {
        "status": "packed",
        "project": project_name,
        "out_path": str(out_path),
        "sealed": sealed,
        "file_count": file_count,
        "bytes": total_bytes,
    }


def _read_manifest(zf: zipfile.ZipFile) -> dict[str, Any] | None:
    try:
        with zf.open(MANIFEST_NAME) as f:
            data: dict[str, Any] = json.loads(f.read().decode("utf-8"))
            return data
    except KeyError:
        return None
    except Exception as exc:
        raise ProjectPackError(f"Corrupt pack manifest: {exc}") from exc


def _validate_member_path(name: str, staging: Path) -> Path:
    """Return the safe resolved destination for a zip member, or raise
    ProjectPackError. Rejects absolute paths, '..' traversal, and any
    path that would resolve outside staging."""
    if os.path.isabs(name) or ".." in Path(name).parts:
        raise ProjectPackError(f"Unsafe path in archive: {name!r}")
    resolved_staging = staging.resolve()
    dest = (staging / name).resolve()
    if dest != resolved_staging and resolved_staging not in dest.parents:
        raise ProjectPackError(f"Path escapes target directory: {name!r}")
    return dest


def _ensure_ancestor_skeleton(ancestor_dir: Path) -> None:
    """Create a minimal project skeleton for an --as parent/child target's
    ancestor segment, anchored at the explicit dir passed in — NOT via
    axon.projects.ensure_project() (which resolves against the
    process-global PROJECTS_ROOT; see _resolve_project_dir's docstring).
    Idempotent: safe to call on an ancestor that already fully exists.
    """
    for d in _SKELETON_DIRS:
        (ancestor_dir / d).mkdir(parents=True, exist_ok=True)
    meta_path = ancestor_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {"name": ancestor_dir.name, "created_at": datetime.now(timezone.utc).isoformat()}
            ),
            encoding="utf-8",
        )


def unpack_project(
    zip_path: Path | str,
    user_dir: Path,
    *,
    as_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Restore a zip produced by :func:`pack_project` into AxonStore.

    Brain-agnostic: the caller owns switching away from the target
    project first if *force* is set and it's currently active — Windows
    file locks would otherwise fail the replace step.

    Returns ``{"status": "unpacked", "project", "sealed", "file_count",
    "manifest"}``.
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise ProjectPackError(f"Pack file does not exist: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        manifest = _read_manifest(zf)
        target_name = as_name or (manifest or {}).get("original_project")
        if not target_name:
            raise ProjectPackError(
                "No target project name: the pack has no manifest and --as/as_name was not given."
            )
        try:
            segments = _parse_name(target_name)
        except ValueError as exc:
            raise ProjectPackError(f"Invalid target project name '{target_name}': {exc}") from exc

        target_dir = _resolve_project_dir(target_name, Path(user_dir))
        if target_dir.exists() and not force:
            raise ProjectPackError(
                f"Project '{target_name}' already exists. Pass force=True to overwrite."
            )

        staging = target_dir.parent / f".{target_dir.name}.unpacking-{secrets.token_hex(6)}"
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

        infos = [i for i in zf.infolist() if i.filename != MANIFEST_NAME]
        # Pass 1: validate every member before writing anything — reject
        # the whole unpack on any violation, never a partial extraction.
        planned: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in infos:
            name = info.filename
            if name.endswith("/"):
                continue
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ProjectPackError(f"Archive contains a symlink entry: {name!r}")
            dest = _validate_member_path(name, staging)
            planned.append((info, dest))

        # Pass 2: extract into staging — never the real target directory.
        try:
            for info, dest in planned:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
            for d in _SKELETON_DIRS:
                (staging / d).mkdir(parents=True, exist_ok=True)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    # Ancestor skeleton for a nested --as parent/child target — every
    # segment except the leaf must exist as a real project before the
    # leaf can be nested under its subs/.
    if len(segments) > 1:
        ancestor_dir = Path(user_dir) / segments[0]
        _ensure_ancestor_skeleton(ancestor_dir)
        for seg in segments[1:-1]:
            ancestor_dir = ancestor_dir / "subs" / seg
            _ensure_ancestor_skeleton(ancestor_dir)

    if target_dir.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(target_dir)
                break
            except PermissionError:
                if attempt == 2:
                    shutil.rmtree(staging, ignore_errors=True)
                    raise
                time.sleep(0.5)
            except FileNotFoundError:
                break
    os.replace(staging, target_dir)

    sealed = bool((manifest or {}).get("sealed", _is_project_sealed(target_dir)))
    return {
        "status": "unpacked",
        "project": target_name,
        "sealed": sealed,
        "file_count": len(planned),
        "manifest": manifest,
    }
