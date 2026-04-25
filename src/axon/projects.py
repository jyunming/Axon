"""


Project management for Axon.


Each project gets its own isolated vector store and BM25 index under:
    ~/.axon/projects/<name>/
        vector_store_data/  — vector store data (tqdb, lancedb, chroma, qdrant)
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


AxonStore — the only storage layout used by Axon


-------------------------------------------------


Each OS user gets a namespace under {axon_store_base}/AxonStore/{username}/ containing:
    store_meta.json  — store-level identity and version metadata
    default/         — the user's default project
    <project>/       — user-created local projects live at the namespace root
    projects/        — compatibility directory reserved for store tooling
    mounts/          — read-only mount descriptors (canonical)
    .shares/         — share key manifests


"""


import hashlib
import json
import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


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


# Keep special AxonStore roots reserved so callers cannot create projects that


# collide with mounted-share routing or legacy default names.


_RESERVED_NAMES: set = {"projects", "mounts", "sharemount", "_default", ".shares"}


def is_reserved_top_level_name(name: str) -> bool:
    """Return True when *name* targets a reserved top-level AxonStore root."""
    segments = [seg for seg in name.split("/") if seg]
    return bool(segments) and segments[0] in _RESERVED_NAMES


# ---------------------------------------------------------------------------


# Namespace ID helpers  (Phase 1)


# ---------------------------------------------------------------------------


def build_project_id(prefix: str = "ns") -> str:
    """Generate a new random namespace ID.
    Format: ``{prefix}_{uuid4_hex}``  — e.g. ``proj_3f2a...``
    Args:
        prefix: Short label for the kind of namespace (``"proj"``, ``"store"``).
    Returns:
        A string of the form ``{prefix}_{uuid4_hex}`` (len = len(prefix) + 33)
        that is unique with overwhelming probability.
    """
    return f"{prefix}_{uuid.uuid4().hex}"


def build_source_id(project_id: str, source_kind: str, canonical_source_locator: str) -> str:
    """Derive a stable source ID from the project + source kind + canonical locator.
    Formula: sha256(project_id | source_kind | canonical_source_locator)[:24]
    prefixed with "src_".
    Args:
        project_id: From meta.json, e.g. "proj_3f2a..."
        source_kind: "file", "url", "code", "text", etc.
        canonical_source_locator: Normalized relative path, canonical URL, or logical key.
            - For files: use the absolute path (normalized with os.path.normpath)
            - For URLs: use the URL as-is
            - For text snippets: use a stable logical key (e.g. content hash)
    Returns:
        A 28-character string: "src_" + first 24 hex chars of sha256.
    """
    raw = f"{project_id}|{source_kind}|{canonical_source_locator}"
    return "src_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def build_chunk_id(
    project_id: str,
    source_id: str,
    subdoc_locator: str,
    chunk_index: int,
    chunk_kind: str = "leaf",
) -> str:
    """Derive a stable globally-unique chunk ID.
    Formula: sha256(project_id | source_id | subdoc_locator | chunk_index | chunk_kind)[:24]
    prefixed with "chk_".
    Args:
        project_id: Project ID from meta.json.
        source_id: Source ID from build_source_id().
        subdoc_locator: Sub-document position, e.g. "page:3", "sheet:Orders", "root", "row:44".
        chunk_index: Zero-based index of this chunk within the subdoc.
        chunk_kind: "leaf", "parent", "raptor_l1", "raptor_l2", "code".
    Returns:
        A 28-character string: "chk_" + first 24 hex chars of sha256.
    """
    raw = f"{project_id}|{source_id}|{subdoc_locator}|{chunk_index}|{chunk_kind}"
    return "chk_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


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
    """Return the absolute path to the project's vector store directory (``vector_store_data/``)."""
    return str(project_dir(name) / "vector_store_data")


def project_bm25_path(name: str) -> str:
    """Return the absolute path to the project's BM25 index directory."""
    return str(project_dir(name) / "bm25_index")


def project_sessions_path(name: str) -> str:
    """Return the absolute path to the project's sessions directory."""
    return str(project_dir(name) / "sessions")


def ensure_project(name: str, description: str = "", security_mode: str | None = None) -> Path:
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


def _ensure_single_project(name: str, description: str, graph_backend: str = "graphrag") -> Path:
    """Create directories and meta.json for exactly one project node.
    If ``meta.json`` already exists but is missing ``project_id``,
    a new one is assigned and the file is updated in-place (migration path).
    ``graph_backend`` is written to meta.json and is immutable once set —
    attempting to change it raises ``ValueError``.
    """
    root = project_dir(name)
    (root / "vector_store_data").mkdir(parents=True, exist_ok=True)
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
                    "project_id": build_project_id("proj"),
                    "graph_backend": graph_backend,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        changed = False
        # Backfill missing project_id for existing projects (Phase 1 migration)
        if "project_id" not in meta:
            meta["project_id"] = build_project_id("proj")
            changed = True
        # Backfill or enforce graph_backend immutability
        if "graph_backend" not in meta:
            meta["graph_backend"] = graph_backend
            changed = True
        elif meta["graph_backend"] != graph_backend:
            raise ValueError(
                f"graph_backend for project '{name}' is immutable: "
                f"stored='{meta['graph_backend']}', requested='{graph_backend}'"
            )
        if changed:
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


def get_project_id(name: str) -> str | None:
    """Return the ``project_id`` from a project's ``meta.json``.
    Returns ``None`` if the project does not exist or has no namespace ID yet.
    Call :func:`ensure_project` first to guarantee the field is present.
    """
    meta_file = project_dir(name) / "meta.json"
    if not meta_file.exists():
        return None
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        val = meta.get("project_id")
        return str(val) if val is not None else None
    except (json.JSONDecodeError, OSError):
        return None


def get_project_graph_backend(name: str) -> str:
    """Return the ``graph_backend`` from a project's ``meta.json``.
    Returns ``"graphrag"`` if the project does not exist, the file is
    unreadable, or the field is absent (legacy project).
    """
    meta_file = project_dir(name) / "meta.json"
    if not meta_file.exists():
        return "graphrag"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        return str(meta.get("graph_backend", "graphrag"))
    except (json.JSONDecodeError, OSError):
        return "graphrag"


def get_store_id(user_dir: Path) -> str | None:
    """Return the ``store_id`` from ``store_meta.json``.
    Returns ``None`` if the file does not exist or is unreadable.
    """
    store_meta = user_dir / "store_meta.json"
    if not store_meta.exists():
        return None
    try:
        meta = json.loads(store_meta.read_text(encoding="utf-8"))
        val = meta.get("store_id")
        return str(val) if val is not None else None
    except (json.JSONDecodeError, OSError):
        return None


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
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        full_name = f"{parent_name}/{entry.name}"
        state = meta.get("maintenance_state", "normal")
        result.append(
            {
                "name": full_name,
                "description": meta.get("description", ""),
                "created_at": meta.get("created_at", ""),
                "path": str(entry),
                "maintenance_state": state if state in _VALID_MAINTENANCE_STATES else "normal",
                "children": _list_sub_projects(entry, full_name),
            }
        )
    result.sort(key=lambda p: p["created_at"], reverse=True)
    return result


_VALID_MAINTENANCE_STATES: frozenset[str] = frozenset({"normal", "draining", "readonly", "offline"})


def get_maintenance_state(name: str) -> str:
    """Return the maintenance state of a project (defaults to 'normal' if unset).
    Args:
        name: Project name (slash-separated for sub-projects).
    Returns:
        One of: 'normal', 'draining', 'readonly', 'offline'.
    """
    meta_path = project_dir(name) / "meta.json"
    if not meta_path.exists():
        return "normal"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        state = data.get("maintenance_state", "normal")
        return state if state in _VALID_MAINTENANCE_STATES else "normal"
    except Exception:
        return "normal"


def set_maintenance_state(name: str, state: str) -> None:
    """Persist a maintenance state to a project's meta.json.
    Args:
        name: Project name (slash-separated for sub-projects).
        state: One of 'normal', 'draining', 'readonly', 'offline'.
    Raises:
        ValueError: If state is not valid or project does not exist.
    """
    if state not in _VALID_MAINTENANCE_STATES:
        raise ValueError(
            f"Invalid maintenance state '{state}'. "
            f"Valid states: {sorted(_VALID_MAINTENANCE_STATES)}"
        )
    meta_path = project_dir(name) / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"Project '{name}' does not exist.")
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Could not read meta.json for '{name}': {exc}") from exc
    old_state = data.get("maintenance_state", "normal")
    data["maintenance_state"] = state
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info(
        "audit: project '%s' maintenance_state %s → %s",
        name,
        old_state,
        state,
    )


def list_projects() -> list[dict]:
    """Return all top-level projects sorted by creation time (newest first).
    Each dict contains: name, description, created_at, path, maintenance_state, children.
    The 'children' list recursively contains sub-project dicts in the same format.
    """
    if not PROJECTS_ROOT.exists():
        return []
    result: list[dict] = []
    for entry in sorted(PROJECTS_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in _RESERVED_NAMES:
            continue
        meta_file = entry / "meta.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        state = meta.get("maintenance_state", "normal")
        result.append(
            {
                "name": entry.name,
                "description": meta.get("description", ""),
                "created_at": meta.get("created_at", ""),
                "path": str(entry),
                "maintenance_state": state if state in _VALID_MAINTENANCE_STATES else "normal",
                "children": _list_sub_projects(entry, entry.name),
            }
        )
    result.sort(key=lambda p: p["created_at"], reverse=True)
    return result


def get_active_project() -> str:
    """Return the name of the currently active project (defaults to 'default')."""
    if _ACTIVE_FILE.exists():
        name = _ACTIVE_FILE.read_text(encoding="utf-8").strip()
        if name:
            return name
    return "default"


def set_active_project(name: str) -> None:
    """Persist the active project name to disk.
    Args:
        name: Project name to set as active.
    """
    try:
        _ACTIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _ACTIVE_FILE.write_text(name, encoding="utf-8")
    except OSError:
        # Non-fatal: isolated/test environments may not have write access to
        # the home directory. The in-memory switch still takes effect.
        pass


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
        except FileNotFoundError:
            # Another process deleted it concurrently — treat as success
            break
    # If this was the active project, reset to default
    if get_active_project() == name:
        set_active_project("default")


def ensure_user_project(user_dir: Path) -> None:
    """Create the standard subdirectories under a user's AxonStore namespace.
    Creates: default/, mounts/, .shares/, store_meta.json, and a compatibility
    projects/ directory reserved for store tooling.
    Safe to call multiple times (idempotent).
    """
    user_dir.mkdir(parents=True, exist_ok=True)
    # ── Standard structure ────────────────────────────────────────────────────
    (user_dir / "projects").mkdir(exist_ok=True)
    (user_dir / "mounts").mkdir(exist_ok=True)
    (user_dir / ".shares").mkdir(exist_ok=True)
    # Default project
    _ensure_single_project_at(user_dir / "default", "default", "Default project")
    # ── Phase 1: store_meta.json ─────────────────────────────────────────────
    store_meta = user_dir / "store_meta.json"
    if not store_meta.exists():
        store_meta.write_text(
            json.dumps(
                {
                    "store_version": 2,
                    "store_scope": "user_scoped",
                    "store_id": build_project_id("store"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _ensure_single_project_at(root: Path, name: str, description: str) -> Path:
    """Create directories and meta.json for a project at an explicit path.
    If ``meta.json`` already exists but is missing ``project_id``,
    a new one is assigned in-place (migration path).
    """
    (root / "vector_store_data").mkdir(parents=True, exist_ok=True)
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
                    "project_id": build_project_id("proj"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if "project_id" not in meta:
            meta["project_id"] = build_project_id("proj")
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------


# Cleanup


# ---------------------------------------------------------------------------


def _remove_share_link(link: Path) -> bool:
    """Remove a legacy ShareMount/ symlink if present.
    Kept for the transitional cleanup pass in ``validate_received_shares()``.
    Returns True if a symlink was removed, False otherwise.
    """
    if link.is_symlink():
        link.unlink()
        return True
    return False


def list_share_mounts(user_dir: Path) -> list[dict]:
    """List all active received share mounts for *user_dir*.
    Reads from ``mounts/`` descriptor files (canonical source of truth).
    Returns list of dicts with: name, target, is_broken, owner, project.
    """
    from axon.mounts import list_mount_descriptors, validate_mount_descriptor

    results = []
    for desc in list_mount_descriptors(user_dir):
        valid, _ = validate_mount_descriptor(desc)
        results.append(
            {
                "name": desc["mount_name"],
                "target": desc.get("target_project_dir", ""),
                "is_broken": not valid,
                "owner": desc.get("owner", ""),
                "project": desc.get("project", ""),
            }
        )
    return results
