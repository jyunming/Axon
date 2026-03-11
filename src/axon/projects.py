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

Maximum hierarchy depth is 3 levels.

The special name "default" is a sentinel that uses the paths from config.yaml.
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
    Used by OpenStudioBrain to apply the projects_root from config.yaml.
    """
    global PROJECTS_ROOT
    PROJECTS_ROOT = Path(path).expanduser()


_SEGMENT_RE: re.Pattern = re.compile(r"^[a-z0-9][a-z0-9_-]{0,49}$")
_MAX_DEPTH: int = 3


class ProjectHasChildrenError(ValueError):
    """Raised when trying to delete a project that still has sub-projects."""


def _parse_name(name: str) -> list[str]:
    """Parse a slash-separated project name into validated segments.

    Returns:
        List of 1–3 segment strings.

    Raises:
        ValueError: If any segment is invalid or depth exceeds _MAX_DEPTH.
    """
    if name == "default":
        return ["default"]
    segments = name.split("/")
    if len(segments) > _MAX_DEPTH:
        raise ValueError(
            f"Project name '{name}' has {len(segments)} levels; " f"maximum depth is {_MAX_DEPTH}."
        )
    for seg in segments:
        if not _SEGMENT_RE.match(seg):
            raise ValueError(
                f"Invalid project name segment '{seg}'. "
                "Use lowercase letters, digits, hyphens, and underscores "
                "(1–50 chars, must start with a letter or digit)."
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
    """Return the absolute path to the project's ChromaDB directory."""
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
    _validate_name(name)
    if name == "default":
        root = project_dir("default")
    else:
        segments = _parse_name(name)
        # Ensure every ancestor exists (without overwriting existing meta.json)
        for depth in range(1, len(segments)):
            ancestor_name = "/".join(segments[:depth])
            _ensure_single_project(ancestor_name, description="")
        root = project_dir(name)
    _ensure_single_project(name, description=description)
    return root


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


def list_descendants(name: str) -> list[str]:
    """Return the full slash-separated names of all descendant projects.

    Traverses at most two levels deep (since max total depth is 3).

    Args:
        name: Slash-separated project name (e.g. 'research').

    Returns:
        Sorted list of descendant names (e.g. ['research/papers', 'research/papers/2024']).
    """
    root = project_dir(name)
    subs_dir = root / "subs"
    if not subs_dir.exists():
        return []

    result: list[str] = []
    for sub_entry in sorted(subs_dir.iterdir()):
        if not sub_entry.is_dir() or not (sub_entry / "meta.json").exists():
            continue
        child_name = f"{name}/{sub_entry.name}"
        result.append(child_name)
        # One more level (grandchildren)
        gc_subs = sub_entry / "subs"
        if gc_subs.exists():
            for gc_entry in sorted(gc_subs.iterdir()):
                if not gc_entry.is_dir() or not (gc_entry / "meta.json").exists():
                    continue
                result.append(f"{child_name}/{gc_entry.name}")
    return result


def has_children(name: str) -> bool:
    """Return True if the project has any sub-projects."""
    return bool(list_descendants(name))


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
    shutil.rmtree(root)
    # If this was the active project, reset to default
    if get_active_project() == name:
        set_active_project("default")
