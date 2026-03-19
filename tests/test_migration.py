"""tests/test_migration.py — Migration utility tests."""
import json


def test_migrate_project_meta_backfills(tmp_path):
    from axon.migration import migrate_project_meta

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(json.dumps({"name": "myproject", "created_at": "2026-01-01"}))
    result = migrate_project_meta(proj)
    assert result["action"] == "backfilled"
    assert result["project_namespace_id"].startswith("proj_")
    # Verify written to disk
    meta = json.loads((proj / "meta.json").read_text())
    assert meta["project_namespace_id"] == result["project_namespace_id"]


def test_migrate_project_meta_already_present(tmp_path):
    from axon.migration import migrate_project_meta

    proj = tmp_path / "myproject"
    proj.mkdir()
    (proj / "meta.json").write_text(
        json.dumps({"name": "myproject", "project_namespace_id": "proj_abc"})
    )
    result = migrate_project_meta(proj)
    assert result["action"] == "already_present"
    assert result["project_namespace_id"] == "proj_abc"


def test_migrate_project_meta_missing_meta(tmp_path):
    from axon.migration import migrate_project_meta

    proj = tmp_path / "myproject"
    proj.mkdir()
    result = migrate_project_meta(proj)
    assert result["action"] == "meta_missing"
    assert result["project_namespace_id"] is None


def test_migrate_projects_root(tmp_path):
    from axon.migration import migrate_projects_root

    for name in ("alpha", "beta"):
        p = tmp_path / name
        p.mkdir()
        (p / "meta.json").write_text(json.dumps({"name": name}))
    results = migrate_projects_root(tmp_path)
    assert len(results) == 2
    assert all(r["action"] == "backfilled" for r in results)


def test_audit_legacy_chunk_ids_no_corpus(tmp_path):
    from axon.migration import audit_legacy_chunk_ids

    proj = tmp_path / "myproject"
    proj.mkdir()
    result = audit_legacy_chunk_ids(proj)
    assert result["total_docs"] == 0
    assert result["legacy_id_count"] == 0


def test_audit_legacy_chunk_ids_detects_basenames(tmp_path):
    from axon.migration import audit_legacy_chunk_ids

    proj = tmp_path / "myproject"
    (proj / "bm25_index").mkdir(parents=True)
    corpus = {
        "overview.txt": {"text": "a"},  # legacy — pure basename, no stable prefix, no _chunk_
        "file_abc123_chunk_0": {"text": "b"},  # stable (new style)
        "readme.md": {"text": "c"},  # legacy — pure basename
    }
    (proj / "bm25_index" / "corpus.json").write_text(json.dumps(corpus))
    result = audit_legacy_chunk_ids(proj)
    assert result["total_docs"] == 3
    assert result["legacy_id_count"] == 2


def test_audit_legacy_chunk_ids_clean(tmp_path):
    from axon.migration import audit_legacy_chunk_ids

    proj = tmp_path / "myproject"
    (proj / "bm25_index").mkdir(parents=True)
    corpus = {
        "file_abc123_chunk_0": {"text": "a"},
        "code_xyz789_chunk_0": {"text": "b"},
    }
    (proj / "bm25_index" / "corpus.json").write_text(json.dumps(corpus))
    result = audit_legacy_chunk_ids(proj)
    assert result["legacy_id_count"] == 0
