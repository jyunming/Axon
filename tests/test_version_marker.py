"""Unit tests for axon.version_marker — share-mount staleness detection."""
from __future__ import annotations

import json
from pathlib import Path

from axon.version_marker import (
    MANIFEST_FILES,
    SCHEMA_VERSION,
    VERSION_MARKER_FILENAME,
    bump,
    is_newer_than,
    read,
    rollup_hashes,
)


def _seed_project(root: Path) -> Path:
    """Create a tiny project layout matching the manifest-file paths."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "bm25_index").mkdir()
    (root / "vector_store_data").mkdir()
    (root / "meta.json").write_text('{"project_id": "p1"}', encoding="utf-8")
    (root / "bm25_index" / ".bm25_log.jsonl").write_text('{"seq":1}\n', encoding="utf-8")
    (root / "vector_store_data" / "manifest.json").write_text('{"d":768,"b":4}', encoding="utf-8")
    return root


class TestRollupHashes:
    def test_hashes_only_existing_manifest_files(self, tmp_path):
        _seed_project(tmp_path)
        out = rollup_hashes(tmp_path)
        # We seeded three of the four canonical files (no dynamic-graph snapshot).
        assert "meta.json" in out
        assert "bm25_index/.bm25_log.jsonl" in out
        assert "vector_store_data/manifest.json" in out
        assert "bm25_index/.dynamic_graph.snapshot.json" not in out
        # Each hash is a 64-hex SHA-256 digest.
        assert all(len(v) == 64 and all(c in "0123456789abcdef" for c in v) for v in out.values())

    def test_empty_project_yields_empty_dict(self, tmp_path):
        assert rollup_hashes(tmp_path) == {}

    def test_change_to_one_file_changes_one_hash(self, tmp_path):
        _seed_project(tmp_path)
        before = rollup_hashes(tmp_path)
        (tmp_path / "meta.json").write_text('{"project_id": "p1-new"}', encoding="utf-8")
        after = rollup_hashes(tmp_path)
        assert after["meta.json"] != before["meta.json"]
        # Other files unchanged.
        assert after["bm25_index/.bm25_log.jsonl"] == before["bm25_index/.bm25_log.jsonl"]


class TestBump:
    def test_first_bump_writes_seq_1(self, tmp_path):
        _seed_project(tmp_path)
        marker = bump(tmp_path)
        assert marker["seq"] == 1
        assert marker["schema_version"] == SCHEMA_VERSION
        assert marker["hash_algo"] == "sha256"
        # File exists and round-trips through read().
        assert (tmp_path / VERSION_MARKER_FILENAME).is_file()
        assert read(tmp_path) == marker

    def test_subsequent_bumps_increment_seq(self, tmp_path):
        _seed_project(tmp_path)
        m1 = bump(tmp_path)
        m2 = bump(tmp_path)
        m3 = bump(tmp_path)
        assert m1["seq"] == 1
        assert m2["seq"] == 2
        assert m3["seq"] == 3

    def test_explicit_seq_override(self, tmp_path):
        _seed_project(tmp_path)
        marker = bump(tmp_path, seq=42)
        assert marker["seq"] == 42

    def test_atomic_write_no_tempfile_left(self, tmp_path):
        _seed_project(tmp_path)
        bump(tmp_path)
        # The .json.tmp staging file must not survive a successful write.
        assert not (tmp_path / "version.json.tmp").exists()

    def test_artifact_hashes_recorded(self, tmp_path):
        _seed_project(tmp_path)
        marker = bump(tmp_path)
        assert "meta.json" in marker["artifacts"]
        assert "bm25_index/.bm25_log.jsonl" in marker["artifacts"]


class TestRead:
    def test_missing_marker_returns_none(self, tmp_path):
        assert read(tmp_path) is None

    def test_corrupt_marker_returns_none(self, tmp_path):
        (tmp_path / VERSION_MARKER_FILENAME).write_text("{not json", encoding="utf-8")
        assert read(tmp_path) is None

    def test_non_dict_marker_returns_none(self, tmp_path):
        (tmp_path / VERSION_MARKER_FILENAME).write_text("[1,2,3]", encoding="utf-8")
        assert read(tmp_path) is None


class TestIsNewerThan:
    def test_no_cached_means_anything_with_marker_is_newer(self):
        assert is_newer_than({"seq": 1, "artifacts": {}}, None) is True

    def test_no_current_returns_false(self):
        assert is_newer_than(None, {"seq": 1, "artifacts": {}}) is False
        assert is_newer_than(None, None) is False

    def test_higher_seq_is_newer(self):
        assert is_newer_than({"seq": 2, "artifacts": {}}, {"seq": 1, "artifacts": {}}) is True

    def test_lower_seq_is_not_newer(self):
        assert is_newer_than({"seq": 1, "artifacts": {}}, {"seq": 2, "artifacts": {}}) is False

    def test_same_seq_diff_artifacts_is_newer(self):
        a = {"seq": 5, "artifacts": {"meta.json": "aaa"}}
        b = {"seq": 5, "artifacts": {"meta.json": "bbb"}}
        assert is_newer_than(b, a) is True

    def test_same_seq_same_artifacts_is_not_newer(self):
        a = {"seq": 5, "artifacts": {"meta.json": "aaa"}}
        b = {"seq": 5, "artifacts": {"meta.json": "aaa"}}
        assert is_newer_than(b, a) is False


class TestRoundTrip:
    """End-to-end: owner bumps after each ingest; grantee reads marker."""

    def test_grantee_observes_owner_seq_advance(self, tmp_path):
        owner_dir = tmp_path / "owner_project"
        _seed_project(owner_dir)

        # Owner bumps; grantee reads. seq=1.
        owner_marker = bump(owner_dir)
        grantee_cached = read(owner_dir)
        assert grantee_cached["seq"] == 1
        assert not is_newer_than(grantee_cached, owner_marker)

        # Owner re-ingests and bumps; grantee re-reads. seq=2 > cached=1.
        (owner_dir / "meta.json").write_text('{"project_id": "p1", "rev": 2}', encoding="utf-8")
        bump(owner_dir)
        latest = read(owner_dir)
        assert is_newer_than(latest, grantee_cached)


class TestManifestFilesConstant:
    def test_known_manifest_files(self):
        # Sanity: drift in this list is API-breaking, lock it down.
        assert "meta.json" in MANIFEST_FILES
        assert "bm25_index/.bm25_log.jsonl" in MANIFEST_FILES
        assert "vector_store_data/manifest.json" in MANIFEST_FILES
        assert "bm25_index/.dynamic_graph.snapshot.json" in MANIFEST_FILES


class TestMarkerJsonShape:
    """The on-disk JSON shape is a compatibility surface for grantees on
    different Axon versions — pin it down."""

    def test_marker_keys_are_stable(self, tmp_path):
        _seed_project(tmp_path)
        bump(tmp_path)
        raw = json.loads((tmp_path / VERSION_MARKER_FILENAME).read_text(encoding="utf-8"))
        assert set(raw.keys()) == {
            "schema_version",
            "seq",
            "generated_at",
            "owner_host",
            "hash_algo",
            "artifacts",
        }
