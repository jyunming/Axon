"""
axon/migration.py

Migration utilities for upgrading existing Axon project data to the
current schema version. Designed to be run once per project after upgrading.

Current migrations:
  - v1_to_v2: assigns project_namespace_id to meta.json if missing
              (backfill is already done automatically by ensure_project;
               this script is for explicit bulk migration and validation)
  - legacy_ids: reports presence of legacy basename-derived chunk IDs
                in vector store metadata (does not rewrite — reingestion required)
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def migrate_project_meta(project_dir: Path) -> dict[str, Any]:
    """Ensure a project directory has project_namespace_id in meta.json.

    Returns a dict with keys:
      - "project": project dir name
      - "action": "backfilled" | "already_present" | "meta_missing"
      - "project_namespace_id": the ID (new or existing), or None if missing
    """
    from axon.projects import build_namespace_id

    meta_file = project_dir / "meta.json"
    if not meta_file.exists():
        return {
            "project": project_dir.name,
            "action": "meta_missing",
            "project_namespace_id": None,
        }

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if "project_namespace_id" in meta:
        return {
            "project": project_dir.name,
            "action": "already_present",
            "project_namespace_id": meta["project_namespace_id"],
        }

    ns_id = build_namespace_id("proj")
    meta["project_namespace_id"] = ns_id
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {
        "project": project_dir.name,
        "action": "backfilled",
        "project_namespace_id": ns_id,
    }


def migrate_projects_root(projects_root: Path) -> list[dict[str, Any]]:
    """Run migrate_project_meta for every project directory under projects_root.

    Walks the top-level directories that contain a meta.json file.
    Does not recurse into subs/ — call ensure_project() for that.

    Returns a list of result dicts from migrate_project_meta.
    """
    results: list[dict[str, Any]] = []
    if not projects_root.exists():
        return results
    for entry in sorted(projects_root.iterdir()):
        if entry.is_dir() and (entry / "meta.json").exists():
            results.append(migrate_project_meta(entry))
    return results


def audit_legacy_chunk_ids(project_dir: Path) -> dict[str, Any]:
    """Check whether a project's BM25 corpus contains legacy basename-derived IDs.

    A legacy ID is one that does NOT start with a known stable prefix
    (file_, json_, html_, docx_, pdf_, epub_, code_, url_) or contain "_chunk_".

    Note: This audit only inspects the BM25 corpus JSON (fast, no vector store access).
    It does NOT rewrite IDs — if legacy IDs are found, the project must be re-ingested.

    Returns:
      - "project": project dir name
      - "bm25_corpus_path": path checked
      - "total_docs": number of docs in corpus
      - "legacy_id_count": number of docs with legacy IDs
      - "sample_legacy_ids": up to 5 example legacy IDs
    """
    bm25_dir = project_dir / "bm25_index"
    canonical_path = bm25_dir / "bm25_corpus.json"
    legacy_path = bm25_dir / "corpus.json"
    if canonical_path.exists():
        bm25_path = canonical_path
    elif legacy_path.exists():
        bm25_path = legacy_path
    else:
        return {
            "project": project_dir.name,
            "bm25_corpus_path": str(canonical_path),
            "total_docs": 0,
            "legacy_id_count": 0,
            "sample_legacy_ids": [],
        }

    corpus = json.loads(bm25_path.read_text(encoding="utf-8"))
    _STABLE_PREFIXES = (
        "file_",
        "json_",
        "html_",
        "docx_",
        "pdf_",
        "epub_",
        "code_",
        "url_",
        "chk_",
        "src_",
    )

    total = len(corpus)
    legacy = [
        doc_id
        for doc_id in corpus
        if not any(doc_id.startswith(p) for p in _STABLE_PREFIXES) and "_chunk_" not in doc_id
    ]
    return {
        "project": project_dir.name,
        "bm25_corpus_path": str(bm25_path),
        "total_docs": total,
        "legacy_id_count": len(legacy),
        "sample_legacy_ids": legacy[:5],
    }


def run_migration(projects_root: Path | str, verbose: bool = True) -> None:
    """Run all migrations for every project under projects_root.

    Suitable for use from the CLI or a one-off script:

        python -c "from axon.migration import run_migration; from pathlib import Path; run_migration(Path.home() / '.axon/projects')"
    """
    root = Path(projects_root)
    logger.info("Migration target: %s", root)
    print(f"Migration target: {root}")

    meta_results = migrate_projects_root(root)
    for r in meta_results:
        action = r["action"]
        ns = r.get("project_namespace_id") or "N/A"
        logger.info("  [%16s] %s  ns=%s", action, r["project"], ns)
        if action == "backfilled":
            print(f"  [backfilled] {r['project']}  ns={ns}")

    if verbose and root.exists():
        logger.info("Legacy ID audit:")
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and (entry / "meta.json").exists():
                audit = audit_legacy_chunk_ids(entry)
                if audit["legacy_id_count"] > 0:
                    logger.warning(
                        "  WARN  %s: %d/%d docs have legacy IDs",
                        entry.name,
                        audit["legacy_id_count"],
                        audit["total_docs"],
                    )
                    for sid in audit["sample_legacy_ids"]:
                        logger.warning("        example: %r", sid)
                else:
                    logger.info(
                        "  OK    %s: %d docs, no legacy IDs", entry.name, audit["total_docs"]
                    )
