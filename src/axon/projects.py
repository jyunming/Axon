"""
Project management for Axon.

Each project gets its own isolated vector store and BM25 index under:
    ~/.axon/projects/<name>/
        chroma_data/   — ChromaDB collection
        bm25_index/    — BM25 JSON corpus
        sessions/      — REPL sessions for this project
        meta.json      — project metadata (name, description, created_at)
        subs/          — sub-projects directory

Sub-projects use nested 'subs/' directories:
    research              → ~/.axon/projects/research/
    research/papers       → ~/.axon/projects/research/subs/papers/
    research/papers/2024  → ~/.axon/projects/research/subs/papers/subs/2024/

Maximum hierarchy depth is 5 levels.

The special name "default" is a sentinel that uses the paths from config.yaml.

AxonStore multi-user shared storage
------------------------------------
When AxonStore mode is active (axon_store_base is set), each OS user gets a
namespace under {axon_store_base}/AxonStore/{username}/ containing:
    ShareMount/  — symlinks to other users' shared projects
    .shares/     — share key manifests
    _default/    — the user's default project
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _resolve_projects_root() -> Path:
    """Return the projects root, honouring AXON_PROJECTS_ROOT if set."""
    env = os.environ.get("AXON_PROJECTS_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".axon" / "projects"


PROJECTS_ROOT: Path = _resolve_projects_root()
_ACTIVE_FILE: Path = Path.home() / ".axon" / ".active_project"


def set_projects_root(path: str | Path) -> None:
    """Override PROJECTS_ROOT at runtime (call before any project operations).

    Priority order: explicit call here > AXON_PROJECTS_ROOT env var > default.
    Used by AxonBrain to apply the projects_root from config.yaml.
    """
    global PROJECTS_ROOT
    PROJECTS_ROOT = Path(path).expanduser()


_SEGMENT_RE: re.Pattern = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}$")
_MAX_DEPTH: int = 5
_RESERVED_NAMES: set = {"sharemount", "_default", ".shares"}


class ProjectHasChildrenError(ValueError):
    """Raised when trying to delete a project that still has sub-projects."""


def _parse_name(name: str) -> list[str]:
    """Parse a slash-separated project name into validated segments.

    Returns:
        List of 1–5 segment strings.

    Raises:
        ValueError: If any segment is invalid or depth exceeds _MAX_DEPTH.
    """
    if name == "default":
        return ["default"]
    segments = name.split("/")
    if len(segments) > _MAX_DEPTH:
        raise ValueError(
            f"Project name '{name}' has {len(segments)} levels; " f"maximum depth is 5."
        )
    for i, seg in enumerate(segments):
        if not _SEGMENT_RE.match(seg):
            raise ValueError(
                f"Invalid project name segment '{seg}'. "
                "Use lowercase letters, digits, hyphens, and underscores "
                "(1–50 chars, must start with a letter or digit)."
            )
        # Check reserved names for top-level segment only
        if i == 0 and seg.lower() in _RESERVED_NAMES:
            raise ValueError(
                f"Project name '{seg}' is reserved and cannot be used as a top-level project name."
            )
    return segments


def _validate_name(name: str) -> None:
    """Raise ValueError if project name (or any segment) is invalid."""
    _parse_name(name)


def project_dir(name: str) -> Path:
    """Return the root directory for a project (may not exist yet).

    Sub-projects are nested with 'subs/' directories:
        research              -> PROJECTS_ROOT/research
        research/papers       -> PROJECTS_ROOT/research/subs/papers
        research/papers/2024  -> PROJECTS_ROOT/research/subs/papers/subs/2024
    """
    if name == "default":
        return PROJECTS_ROOT / "default"
    segments = _parse_name(name)
    path = PROJECTS_ROOT / segments[0]
    for seg in segments[1:]:
        path = path / "subs" / seg
    return path


def project_vector_path(name: str) -> str:
    """Return the absolute path to the project's vector store directory (``chroma_data/``)."""
    return str(project_dir(name) / "chroma_data")


def project_bm25_path(name: str) -> str:
    """Return the absolute path to the project's BM25 index directory."""
    return str(project_dir(name) / "bm25_index")


def project_sessions_path(name: str) -> str:
    """Return the absolute path to the project's sessions directory."""
    return str(project_dir(name) / "sessions")


def ensure_project(name: str, description: str = "") -> Path:
    """Create project directory structure and meta.json if they don't exist.

    For sub-projects, all ancestor projects are also created automatically
    (with empty descriptions) so the hierarchy is always consistent.

    Args:
        name: Project name, optionally slash-separated (e.g. 'research/papers').
        description: Optional human-readable description for the target project.

    Returns:
        Path to the target project root directory.
    """
    # _parse_name validates and returns segments in one call — no double-parse.
    segments = _parse_name(name)
    if name != "default":
        # Ensure every ancestor exists (without overwriting existing meta.json)
        for depth in range(1, len(segments)):
            ancestor_name = "/".join(segments[:depth])
            _ensure_single_project(ancestor_name, description="")
    _ensure_single_project(name, description=description)
    return project_dir(name)


def _ensure_single_project(name: str, description: str) -> Path:
    """Create directories and meta.json for exactly one project node."""
    root = project_dir(name)
    (root / "chroma_data").mkdir(parents=True, exist_ok=True)
    (root / "bm25_index").mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    meta_file = root / "meta.json"
    if not meta_file.exists():
        meta_file.write_text(
            json.dumps(
                {
                    "name": name,
                    "description": description,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
    return root


def list_descendants(name: str, visited: set[str] | None = None) -> list[str]:
    """Return the full slash-separated names of all descendant projects.

    Uses a recursive DFS that supports up to _MAX_DEPTH levels and includes
    cycle detection via Path.resolve() to guard against symlinks in subs/.

    Args:
        name: Slash-separated project name (e.g. 'research').
        visited: Set of resolved path strings already seen (cycle detection).

    Returns:
        Sorted list of descendant names (e.g. ['research/papers', 'research/papers/2024']).
    """
    if visited is None:
        visited = set()

    root = project_dir(name)
    resolved_root = str(root.resolve())
    if resolved_root in visited:
        return []
    visited.add(resolved_root)

    subs_dir = root / "subs"
    if not subs_dir.exists():
        return []

    result: list[str] = []
    for sub_entry in sorted(subs_dir.iterdir()):
        # Skip symlinks in subs/ — subs should never contain symlinks
        if sub_entry.is_symlink():
            continue
        if not sub_entry.is_dir() or not (sub_entry / "meta.json").exists():
            continue
        child_name = f"{name}/{sub_entry.name}"
        result.append(child_name)
        # Recurse for deeper levels
        result.extend(list_descendants(child_name, visited=visited))
    return result


def has_children(name: str) -> bool:
    """Return True if the project has any direct sub-projects.

    Short-circuits on the first child found — does not traverse the full tree.
    """
    subs_dir = project_dir(name) / "subs"
    if not subs_dir.exists():
        return False
    return any(e.is_dir() and (e / "meta.json").exists() for e in subs_dir.iterdir())


def _list_sub_projects(parent_dir: Path, parent_name: str) -> list[dict]:
    """Recursively list sub-project dicts under a parent directory."""
    subs_dir = parent_dir / "subs"
    if not subs_dir.exists():
        return []
    result: list[dict] = []
    for entry in sorted(subs_dir.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            meta = {}
        full_name = f"{parent_name}/{entry.name}"
        result.append(
            {
                "name": full_name,
                "description": meta.get("description", ""),
                "created_at": meta.get("created_at", ""),
                "path": str(entry),
                "children": _list_sub_projects(entry, full_name),
            }
        )
    result.sort(key=lambda p: p["created_at"], reverse=True)
    return result


def list_projects() -> list[dict]:
    """Return all top-level projects sorted by creation time (newest first).

    Each dict contains: name, description, created_at, path, children.
    The 'children' list recursively contains sub-project dicts in the same format.
    """
    if not PROJECTS_ROOT.exists():
        return []
    result: list[dict] = []
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text())
        except Exception:
            meta = {}
        result.append(
            {
                "name": entry.name,
                "description": meta.get("description", ""),
                "created_at": meta.get("created_at", ""),
                "path": str(entry),
                "children": _list_sub_projects(entry, entry.name),
            }
        )
    result.sort(key=lambda p: p["created_at"], reverse=True)
    return result


def get_active_project() -> str:
    """Return the name of the currently active project (defaults to 'default')."""
    if _ACTIVE_FILE.exists():
        name = _ACTIVE_FILE.read_text().strip()
        if name:
            return name
    return "default"


def set_active_project(name: str) -> None:
    """Persist the active project name to disk.

    Args:
        name: Project name to set as active.
    """
    _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_FILE.write_text(name)


def delete_project(name: str) -> None:
    """Delete a project and all its data.

    Args:
        name: Project name to delete.

    Raises:
        ValueError: If trying to delete 'default' or a non-existent project.
        ProjectHasChildrenError: If the project still has sub-projects.
    """
    if name == "default":
        raise ValueError("Cannot delete the 'default' project.")
    _validate_name(name)
    root = project_dir(name)
    if not root.exists():
        raise ValueError(f"Project '{name}' does not exist.")
    children = list_descendants(name)
    if children:
        raise ProjectHasChildrenError(
            f"Project '{name}' has sub-projects: {', '.join(children)}. "
            "Delete the sub-projects first."
        )
    import time

    for attempt in range(3):
        try:
            shutil.rmtree(root)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.5)
    # If this was the active project, reset to default
    if get_active_project() == name:
        set_active_project("default")


def ensure_user_namespace(user_dir: Path) -> None:
    """Create the standard subdirectories under a user's AxonStore namespace.

    Creates: ShareMount/, .shares/, and _default project.
    Safe to call multiple times (idempotent).
    """
    (user_dir / "ShareMount").mkdir(parents=True, exist_ok=True)
    (user_dir / ".shares").mkdir(parents=True, exist_ok=True)
    # _default project
    _ensure_single_project_at(user_dir / "_default", "_default", "Default project")


def _ensure_single_project_at(root: Path, name: str, description: str) -> Path:
    """Create directories and meta.json for a project at an explicit path."""
    (root / "chroma_data").mkdir(parents=True, exist_ok=True)
    (root / "bm25_index").mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    meta_file = root / "meta.json"
    if not meta_file.exists():
        meta_file.write_text(
            json.dumps(
                {
                    "name": name,
                    "description": description,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
        )
    return root


def _make_share_link(target: Path, link: Path) -> None:
    """Create a symlink from link -> target (Linux only).

    Raises:
        OSError: If symlink creation fails.
        NotImplementedError: If called on non-Linux platform.
    """
    import sys

    if sys.platform != "linux":
        raise NotImplementedError("AxonStore sharing is currently only supported on Linux.")
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    os.symlink(target.resolve(), link)


def _remove_share_link(link: Path) -> bool:
    """Remove a symlink from ShareMount/. Returns True if removed, False if not found."""
    if link.is_symlink():
        link.unlink()
        return True
    return False


def list_share_mounts(user_dir: Path) -> list[dict]:
    """List all symlinks under user_dir/ShareMount/.

    Returns list of dicts with: name, target, is_broken, owner, project.
    """
    mount_dir = user_dir / "ShareMount"
    if not mount_dir.exists():
        return []
    results = []
    for entry in sorted(mount_dir.iterdir()):
        if not entry.is_symlink():
            continue
        raw_target = os.readlink(str(entry))
        is_broken = not Path(raw_target).exists()
        parts = entry.name.split("_", 1)
        owner = parts[0] if len(parts) == 2 else ""
        project = parts[1] if len(parts) == 2 else entry.name
        results.append(
            {
                "name": entry.name,
                "target": raw_target,
                "is_broken": is_broken,
                "owner": owner,
                "project": project,
            }
        )
    return results
