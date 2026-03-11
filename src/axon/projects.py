"""
Project management for Axon.

Each project gets its own isolated vector store and BM25 index under:
    ~/.axon/projects/<name>/
        chroma_data/   — ChromaDB collection
        bm25_index/    — BM25 JSON corpus
        meta.json      — project metadata (name, description, created_at)

REPL sessions are stored globally at ~/.axon/sessions/ (not per-project).
The special name "default" is a sentinel that uses the paths from config.yaml.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECTS_ROOT: Path = Path.home() / ".axon" / "projects"
_ACTIVE_FILE: Path = Path.home() / ".axon" / ".active_project"
_NAME_RE: re.Pattern = re.compile(r'^[a-z0-9][a-z0-9_-]{0,49}$')


def _validate_name(name: str) -> None:
    """Raise ValueError if project name is invalid."""
    if name == "default":
        return
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid project name '{name}'. "
            "Use lowercase letters, digits, hyphens, and underscores (1–50 chars, start with letter/digit)."
        )


def project_dir(name: str) -> Path:
    """Return the root directory for a project (may not exist yet)."""
    return PROJECTS_ROOT / name


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

    Args:
        name: Project name (validated).
        description: Optional human-readable description.

    Returns:
        Path to the project root directory.
    """
    _validate_name(name)
    root = project_dir(name)
    (root / "chroma_data").mkdir(parents=True, exist_ok=True)
    (root / "bm25_index").mkdir(parents=True, exist_ok=True)
    (root / "sessions").mkdir(parents=True, exist_ok=True)
    meta_file = root / "meta.json"
    if not meta_file.exists():
        meta_file.write_text(json.dumps({
            "name": name,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2))
    return root


def list_projects() -> List[dict]:
    """Return all projects sorted by creation time (newest first).

    Returns:
        List of dicts: name, description, created_at, path.
    """
    if not PROJECTS_ROOT.exists():
        return []
    result = []
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        meta_file = entry / "meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
            except Exception:
                meta = {}
        else:
            meta = {}
        result.append({
            "name": entry.name,
            "description": meta.get("description", ""),
            "created_at": meta.get("created_at", ""),
            "path": str(entry),
        })
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
    """
    if name == "default":
        raise ValueError("Cannot delete the 'default' project.")
    _validate_name(name)
    root = project_dir(name)
    if not root.exists():
        raise ValueError(f"Project '{name}' does not exist.")
    shutil.rmtree(root)
    # If this was the active project, reset to default
    if get_active_project() == name:
        set_active_project("default")
