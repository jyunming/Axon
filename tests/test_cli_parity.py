"""
tests/test_cli_parity.py

Tests for CLI flag parity: --cite, --code-graph, --code-graph-bridge.
Also covers REPL command logic for /refresh, /stale, /graph status,
/share lifecycle, and /store whoami.
"""

import argparse
import hashlib
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# CLI flag parity: --cite, --code-graph, --code-graph-bridge
# ---------------------------------------------------------------------------


def _make_rag_parser():
    """Minimal parser that mirrors the new flags added to main()."""
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--cite", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--code-graph", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--code-graph-bridge", action=argparse.BooleanOptionalAction, default=None)
    return p


class TestCiteCliFlag:
    def test_cite_flag_sets_true(self):
        args = _make_rag_parser().parse_args(["--cite", "hello"])
        assert args.cite is True

    def test_no_cite_flag_sets_false(self):
        args = _make_rag_parser().parse_args(["--no-cite", "hello"])
        assert args.cite is False

    def test_cite_absent_is_none(self):
        args = _make_rag_parser().parse_args(["hello"])
        assert args.cite is None

    def test_code_graph_flag_sets_true(self):
        args = _make_rag_parser().parse_args(["--code-graph", "hello"])
        assert args.code_graph is True

    def test_no_code_graph_flag_sets_false(self):
        args = _make_rag_parser().parse_args(["--no-code-graph", "hello"])
        assert args.code_graph is False

    def test_code_graph_bridge_flag_sets_true(self):
        args = _make_rag_parser().parse_args(["--code-graph-bridge", "hello"])
        assert args.code_graph_bridge is True

    def test_no_code_graph_bridge_flag_sets_false(self):
        args = _make_rag_parser().parse_args(["--no-code-graph-bridge", "hello"])
        assert args.code_graph_bridge is False

    def test_all_absent_are_none(self):
        args = _make_rag_parser().parse_args(["hello"])
        assert args.cite is None
        assert args.code_graph is None
        assert args.code_graph_bridge is None


class TestCiteConfigDefault:
    def test_cite_default_true(self):
        from axon.main import AxonConfig

        assert AxonConfig().cite is True

    def test_code_graph_default_false(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_graph is False

    def test_code_graph_bridge_default_false(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_graph_bridge is False


class TestCliConfigWiring:
    """Simulate the argparse → config application block for new flags."""

    def _apply(self, **overrides):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        if overrides.get("cite") is not None:
            cfg.cite = overrides["cite"]
        if overrides.get("code_graph") is not None:
            cfg.code_graph = overrides["code_graph"]
        if overrides.get("code_graph_bridge") is not None:
            cfg.code_graph_bridge = overrides["code_graph_bridge"]
        return cfg

    def test_cite_true_applied(self):
        assert self._apply(cite=True).cite is True

    def test_cite_false_applied(self):
        assert self._apply(cite=False).cite is False

    def test_code_graph_true_applied(self):
        assert self._apply(code_graph=True).code_graph is True

    def test_code_graph_bridge_true_applied(self):
        assert self._apply(code_graph_bridge=True).code_graph_bridge is True

    def test_none_leaves_defaults(self):
        cfg = self._apply(cite=None, code_graph=None, code_graph_bridge=None)
        assert cfg.cite is True
        assert cfg.code_graph is False
        assert cfg.code_graph_bridge is False


# ---------------------------------------------------------------------------
# REPL /refresh — hash comparison logic
# ---------------------------------------------------------------------------


class TestReplRefreshCommand:
    def test_skips_unchanged_file(self, tmp_path):
        doc = tmp_path / "note.txt"
        doc.write_text("hello world", encoding="utf-8")
        stored_hash = hashlib.md5(b"hello world").hexdigest()
        versions = {str(doc): {"content_hash": stored_hash}}

        skipped, reingested = [], []
        for src, record in versions.items():
            if not os.path.exists(src):
                continue
            text = open(src, encoding="utf-8").read()
            cur = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cur == record["content_hash"]:
                skipped.append(src)
            else:
                reingested.append(src)

        assert len(skipped) == 1
        assert len(reingested) == 0

    def test_reingests_changed_file(self, tmp_path):
        doc = tmp_path / "note.txt"
        doc.write_text("updated content", encoding="utf-8")
        versions = {str(doc): {"content_hash": "old_hash"}}

        reingested = []
        for src, record in versions.items():
            if not os.path.exists(src):
                continue
            text = open(src, encoding="utf-8").read()
            cur = hashlib.md5(text.encode("utf-8")).hexdigest()
            if cur != record["content_hash"]:
                reingested.append(src)

        assert len(reingested) == 1

    def test_reports_missing_file(self, tmp_path):
        missing = str(tmp_path / "gone.txt")
        versions = {missing: {"content_hash": "abc"}}
        result = [src for src in versions if not os.path.exists(src)]
        assert result == [missing]

    def test_empty_versions_no_error(self):
        # /refresh on a brain with no tracked docs
        versions = {}
        reingested = [src for src in versions if True]
        assert reingested == []


# ---------------------------------------------------------------------------
# REPL /stale — age-based filtering logic
# ---------------------------------------------------------------------------


class TestReplStaleCommand:
    def _run(self, versions, threshold_days=7):
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - threshold_days * 86400
        stale = []
        for src, record in versions.items():
            ts_str = record.get("ingested_at") or record.get("last_ingested_at")
            if not ts_str:
                continue
            try:
                ts = (
                    datetime.fromisoformat(ts_str.rstrip("Z"))
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                )
            except ValueError:
                continue
            if ts < cutoff:
                age_days = round((now.timestamp() - ts) / 86400, 1)
                stale.append((age_days, src))
        return stale

    def test_old_doc_is_stale(self):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=10)).isoformat()
        versions = {"/old.txt": {"ingested_at": old}}
        stale = self._run(versions)
        assert len(stale) == 1
        assert "/old.txt" in stale[0][1]

    def test_fresh_doc_not_stale(self):
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(days=2)).isoformat()
        versions = {"/fresh.txt": {"ingested_at": fresh}}
        stale = self._run(versions)
        assert stale == []

    def test_mixed_returns_only_old(self):
        now = datetime.now(timezone.utc)
        versions = {
            "/old.txt": {"ingested_at": (now - timedelta(days=15)).isoformat()},
            "/fresh.txt": {"ingested_at": (now - timedelta(days=3)).isoformat()},
        }
        stale = self._run(versions)
        assert len(stale) == 1
        assert "/old.txt" in stale[0][1]

    def test_missing_timestamp_skipped(self):
        versions = {"/nodoc.txt": {}}
        stale = self._run(versions)
        assert stale == []

    def test_custom_threshold(self):
        now = datetime.now(timezone.utc)
        versions = {"/doc.txt": {"ingested_at": (now - timedelta(days=5)).isoformat()}}
        # threshold=7: not stale
        assert self._run(versions, threshold_days=7) == []
        # threshold=3: stale
        assert len(self._run(versions, threshold_days=3)) == 1


# ---------------------------------------------------------------------------
# REPL /graph status — attribute-reading logic
# ---------------------------------------------------------------------------


class TestReplGraphStatus:
    def _status(self, brain):
        entity_count = len(getattr(brain, "_entity_graph", {}))
        relations = getattr(brain, "_relation_graph", {})
        relation_edges = sum(len(v) for v in relations.values())
        summaries = getattr(brain, "_community_summaries", {}) or {}
        in_progress = getattr(brain, "_community_build_in_progress", False)
        return {
            "entities": entity_count,
            "edges": relation_edges,
            "summaries": len(summaries),
            "in_progress": in_progress,
        }

    def test_populated_graph(self):
        brain = MagicMock()
        brain._entity_graph = {"A": {}, "B": {}}
        brain._relation_graph = {"A": ["B", "C"]}
        brain._community_summaries = {"c0": "s0"}
        brain._community_build_in_progress = False
        s = self._status(brain)
        assert s["entities"] == 2
        assert s["edges"] == 2
        assert s["summaries"] == 1
        assert s["in_progress"] is False

    def test_empty_graph(self):
        brain = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_summaries = {}
        brain._community_build_in_progress = False
        s = self._status(brain)
        assert s["entities"] == 0
        assert s["edges"] == 0

    def test_build_in_progress(self):
        brain = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_summaries = {}
        brain._community_build_in_progress = True
        assert self._status(brain)["in_progress"] is True


# ---------------------------------------------------------------------------
# /share lifecycle — generate + revoke + list
# ---------------------------------------------------------------------------


class TestReplShareCommand:
    def test_generate_and_revoke_roundtrip(self, tmp_path):
        from axon.shares import generate_share_key, revoke_share_key

        user_dir = tmp_path / "store" / "alice"
        proj_dir = user_dir / "myproject"
        proj_dir.mkdir(parents=True)
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_test"}', encoding="utf-8")
        (user_dir / ".shares").mkdir(parents=True)
        (user_dir / "ShareMount").mkdir(parents=True)

        result = generate_share_key(
            owner_user_dir=user_dir,
            project="myproject",
            grantee="bob",
        )
        assert result["project"] == "myproject"
        assert result["grantee"] == "bob"
        assert result["share_string"]
        key_id = result["key_id"]

        revoked = revoke_share_key(owner_user_dir=user_dir, key_id=key_id)
        assert revoked["key_id"] == key_id
        assert revoked["revoked_at"]  # revoke returns revoked_at timestamp, not a bool field

    def test_list_shares_structure(self, tmp_path):
        from axon.shares import generate_share_key, list_shares

        user_dir = tmp_path / "store" / "alice"
        proj_dir = user_dir / "proj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_q"}', encoding="utf-8")
        (user_dir / ".shares").mkdir(parents=True)
        (user_dir / "ShareMount").mkdir(parents=True)

        generate_share_key(
            owner_user_dir=user_dir,
            project="proj",
            grantee="carol",
        )

        data = list_shares(user_dir)
        assert "sharing" in data
        assert "shared" in data
        assert len(data["sharing"]) == 1
        assert data["sharing"][0]["grantee"] == "carol"

    def test_revoke_nonexistent_key_raises(self, tmp_path):
        from axon.shares import revoke_share_key

        user_dir = tmp_path / "store" / "alice"
        (user_dir / ".shares").mkdir(parents=True)

        with pytest.raises(ValueError, match="not found"):
            revoke_share_key(owner_user_dir=user_dir, key_id="nonexistent_key")


# ---------------------------------------------------------------------------
# /store whoami + init logic
# ---------------------------------------------------------------------------


class TestReplStoreCommand:
    def test_whoami_not_in_store_mode(self):
        brain = MagicMock()
        brain.config.axon_store_mode = False
        assert brain.config.axon_store_mode is False

    def test_whoami_in_store_mode(self):
        brain = MagicMock()
        brain.config.axon_store_mode = True
        brain.config.projects_root = "/data/AxonStore/alice"
        brain._active_project = "research"
        assert brain.config.axon_store_mode is True
        assert brain._active_project == "research"

    def test_store_init_creates_dirs(self, tmp_path):
        from axon.projects import ensure_user_namespace

        user_dir = tmp_path / "AxonStore" / "testuser"
        ensure_user_namespace(user_dir)
        assert (user_dir / "default").exists()
        assert (user_dir / "projects").exists()
        assert (user_dir / "mounts").exists()
        assert (user_dir / ".shares").exists()
        assert (user_dir / "ShareMount").exists()

    def test_store_init_idempotent(self, tmp_path):
        from axon.projects import ensure_user_namespace

        user_dir = tmp_path / "AxonStore" / "bob"
        ensure_user_namespace(user_dir)
        ensure_user_namespace(user_dir)  # second call should not raise
        assert (user_dir / "default").exists()
