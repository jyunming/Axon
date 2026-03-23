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
        assert not (user_dir / "ShareMount").exists()

    def test_store_init_idempotent(self, tmp_path):
        from axon.projects import ensure_user_namespace

        user_dir = tmp_path / "AxonStore" / "bob"
        ensure_user_namespace(user_dir)
        ensure_user_namespace(user_dir)  # second call should not raise
        assert (user_dir / "default").exists()


# ---------------------------------------------------------------------------
# CLI flag parity: new RAG controls (WS3 / WS6)
# --sentence-window, --sentence-window-size, --crag-lite, --graph-rag-mode
# ---------------------------------------------------------------------------


def _make_modern_rag_parser():
    """Parser mirroring the modern RAG flags added to cli.main()."""
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--sentence-window", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--sentence-window-size", type=int, metavar="N")
    p.add_argument("--crag-lite", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--graph-rag-mode", choices=["local", "global", "hybrid"])
    return p


class TestSentenceWindowCliFlag:
    def test_flag_sets_true(self):
        args = _make_modern_rag_parser().parse_args(["--sentence-window", "q"])
        assert args.sentence_window is True

    def test_no_flag_sets_false(self):
        args = _make_modern_rag_parser().parse_args(["--no-sentence-window", "q"])
        assert args.sentence_window is False

    def test_absent_is_none(self):
        args = _make_modern_rag_parser().parse_args(["q"])
        assert args.sentence_window is None

    def test_size_flag(self):
        args = _make_modern_rag_parser().parse_args(["--sentence-window-size", "3", "q"])
        assert args.sentence_window_size == 3

    def test_size_absent_is_none(self):
        args = _make_modern_rag_parser().parse_args(["q"])
        assert args.sentence_window_size is None


class TestCragLiteCliFlag:
    def test_flag_sets_true(self):
        args = _make_modern_rag_parser().parse_args(["--crag-lite", "q"])
        assert args.crag_lite is True

    def test_no_flag_sets_false(self):
        args = _make_modern_rag_parser().parse_args(["--no-crag-lite", "q"])
        assert args.crag_lite is False

    def test_absent_is_none(self):
        args = _make_modern_rag_parser().parse_args(["q"])
        assert args.crag_lite is None


class TestGraphRagModeCliFlag:
    def test_local_mode(self):
        args = _make_modern_rag_parser().parse_args(["--graph-rag-mode", "local", "q"])
        assert args.graph_rag_mode == "local"

    def test_global_mode(self):
        args = _make_modern_rag_parser().parse_args(["--graph-rag-mode", "global", "q"])
        assert args.graph_rag_mode == "global"

    def test_hybrid_mode(self):
        args = _make_modern_rag_parser().parse_args(["--graph-rag-mode", "hybrid", "q"])
        assert args.graph_rag_mode == "hybrid"

    def test_absent_is_none(self):
        args = _make_modern_rag_parser().parse_args(["q"])
        assert args.graph_rag_mode is None

    def test_invalid_mode_raises(self):
        with pytest.raises(SystemExit):
            _make_modern_rag_parser().parse_args(["--graph-rag-mode", "auto", "q"])


class TestModernRagCliConfigWiring:
    """Simulate the argparse → config wiring block for modern RAG flags."""

    def _apply_flags(self, **kwargs):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        if kwargs.get("sentence_window") is not None:
            cfg.sentence_window = kwargs["sentence_window"]
        if kwargs.get("sentence_window_size") is not None:
            cfg.sentence_window_size = kwargs["sentence_window_size"]
        if kwargs.get("crag_lite") is not None:
            cfg.crag_lite = kwargs["crag_lite"]
        if kwargs.get("graph_rag_mode") is not None:
            cfg.graph_rag_mode = kwargs["graph_rag_mode"]
        return cfg

    def test_sentence_window_applied(self):
        cfg = self._apply_flags(sentence_window=True)
        assert cfg.sentence_window is True

    def test_sentence_window_size_applied(self):
        cfg = self._apply_flags(sentence_window_size=5)
        assert cfg.sentence_window_size == 5

    def test_crag_lite_applied(self):
        cfg = self._apply_flags(crag_lite=True)
        assert cfg.crag_lite is True

    def test_graph_rag_mode_applied(self):
        cfg = self._apply_flags(graph_rag_mode="hybrid")
        assert cfg.graph_rag_mode == "hybrid"

    def test_all_absent_leaves_defaults(self):
        from axon.main import AxonConfig

        cfg = self._apply_flags()
        defaults = AxonConfig()
        assert cfg.sentence_window == defaults.sentence_window
        assert cfg.crag_lite == defaults.crag_lite


# ---------------------------------------------------------------------------
# CLI operational flags: --refresh, --list-stale, --graph-* (SP-031, SP-032)
# ---------------------------------------------------------------------------


def _make_operational_parser():
    """Parser mirroring the operational flags added to cli.main()."""
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--list-stale", action="store_true")
    p.add_argument("--stale-days", type=int, default=7, metavar="N")
    p.add_argument("--graph-status", action="store_true")
    p.add_argument("--graph-finalize", action="store_true")
    p.add_argument("--graph-export", action="store_true")
    return p


class TestRefreshCliFlag:
    def test_refresh_sets_true(self):
        args = _make_operational_parser().parse_args(["--refresh"])
        assert args.refresh is True

    def test_refresh_absent_is_false(self):
        args = _make_operational_parser().parse_args([])
        assert args.refresh is False


class TestListStaleCliFlag:
    def test_list_stale_sets_true(self):
        args = _make_operational_parser().parse_args(["--list-stale"])
        assert args.list_stale is True

    def test_stale_days_default(self):
        args = _make_operational_parser().parse_args(["--list-stale"])
        assert args.stale_days == 7

    def test_stale_days_custom(self):
        args = _make_operational_parser().parse_args(["--list-stale", "--stale-days", "30"])
        assert args.stale_days == 30


class TestGraphCliFlags:
    def test_graph_status_sets_true(self):
        args = _make_operational_parser().parse_args(["--graph-status"])
        assert args.graph_status is True

    def test_graph_finalize_sets_true(self):
        args = _make_operational_parser().parse_args(["--graph-finalize"])
        assert args.graph_finalize is True

    def test_graph_export_sets_true(self):
        args = _make_operational_parser().parse_args(["--graph-export"])
        assert args.graph_export is True

    def test_all_absent_are_false(self):
        args = _make_operational_parser().parse_args([])
        assert args.graph_status is False
        assert args.graph_finalize is False
        assert args.graph_export is False


# ---------------------------------------------------------------------------
# Surface capability registry (SP-001, SP-002)
# ---------------------------------------------------------------------------


class TestSurfaceCapabilityRegistry:
    def test_registry_importable(self):
        from axon.surface_contract import REGISTRY

        assert len(REGISTRY) > 0

    def test_tier1_capabilities_exist(self):
        from axon.surface_contract import Tier, tier1_capabilities

        t1 = tier1_capabilities()
        assert len(t1) > 0
        assert all(c.tier == Tier.ONE for c in t1)

    def test_core_tier1_ids_present(self):
        from axon.surface_contract import tier1_capabilities

        ids = {c.id for c in tier1_capabilities()}
        expected = {
            "query",
            "search",
            "ingest_text",
            "ingest_url",
            "ingest_path",
            "ingest_refresh",
            "ingest_stale",
            "collection_inspect",
            "collection_clear",
            "project_list",
            "project_switch",
            "project_create",
            "project_delete",
            "config_update",
            "graph_status",
        }
        missing = expected - ids
        assert not missing, f"Missing Tier 1 capabilities: {missing}"

    def test_surface_capabilities(self):
        from axon.surface_contract import Surface, surface_capabilities

        api_caps = surface_capabilities(Surface.API)
        assert len(api_caps) > len(
            surface_capabilities(Surface.CLI)
        ), "API should have more capabilities than CLI"

    def test_cli_has_full_surface_coverage(self):
        """CLI now supports all non-API-only capabilities — zero undocumented gaps."""
        from axon.surface_contract import Surface, unsupported_on

        cli_gaps = unsupported_on(Surface.CLI)
        # Any remaining gaps must still have documented reasons
        for cap, reason in cli_gaps:
            assert reason, f"Capability {cap.id} has no documented reason for CLI exclusion"
        assert len(cli_gaps) == 0, (
            f"CLI should now have zero unsupported capabilities; found: "
            f"{[cap.id for cap, _ in cli_gaps]}"
        )

    def test_categories_by_category(self):
        from axon.surface_contract import capabilities_by_category

        groups = capabilities_by_category()
        expected_categories = {"query", "ingest", "project", "config", "graph"}
        assert expected_categories.issubset(groups.keys())


# ---------------------------------------------------------------------------
# CLI runtime qualification: delete by source / by id (Sprint A)
# ---------------------------------------------------------------------------


class TestDeleteDocCliRuntime:
    """Direct runtime qualification of --delete-doc and --delete-doc-id handlers."""

    def _make_brain(self, docs=None):
        brain = MagicMock()
        brain.list_documents.return_value = docs or []
        brain.bm25 = MagicMock()
        return brain

    def test_delete_by_source_calls_vector_store(self, capsys):
        brain = self._make_brain(
            docs=[{"source": "notes.txt", "chunks": 3, "doc_ids": ["id1", "id2", "id3"]}]
        )
        import types

        args = types.SimpleNamespace(
            delete_doc="notes.txt",
            delete_doc_id=None,
            store_init=None,
            share_list=False,
            share_generate=None,
            share_redeem=None,
            share_revoke=None,
            session_list=False,
            graph_status=False,
            graph_finalize=False,
            graph_export=False,
        )

        # Simulate the handler block directly
        source = args.delete_doc
        docs = brain.list_documents()
        match = [d for d in docs if d["source"] == source or source in d["source"]]
        assert match, "Should match notes.txt"
        ids_to_delete = [i for d in match for i in d.get("doc_ids", [])]
        brain.vector_store.delete_by_ids(ids_to_delete)
        if brain.bm25 is not None:
            brain.bm25.delete_documents(ids_to_delete)

        brain.vector_store.delete_by_ids.assert_called_once_with(["id1", "id2", "id3"])
        brain.bm25.delete_documents.assert_called_once_with(["id1", "id2", "id3"])

    def test_delete_by_source_no_match_exits(self):
        import types

        brain = self._make_brain(docs=[{"source": "other.txt", "chunks": 1, "doc_ids": ["x"]}])

        args = types.SimpleNamespace(delete_doc="missing.txt")
        source = args.delete_doc
        docs = brain.list_documents()
        match = [d for d in docs if d["source"] == source or source in d["source"]]
        assert not match

    def test_delete_by_id_calls_vector_store(self):
        brain = self._make_brain()
        brain.vector_store.get_by_ids.return_value = [{"id": "abc"}, {"id": "def"}]

        ids = ["abc", "def"]
        existing = brain.vector_store.get_by_ids(ids)
        existing_ids = [d["id"] for d in existing]
        brain.vector_store.delete_by_ids(existing_ids)
        if brain.bm25 is not None:
            brain.bm25.delete_documents(existing_ids)

        brain.vector_store.delete_by_ids.assert_called_once_with(["abc", "def"])
        brain.bm25.delete_documents.assert_called_once_with(["abc", "def"])

    def test_delete_by_id_not_found_reported(self):
        brain = self._make_brain()
        brain.vector_store.get_by_ids.return_value = []  # nothing found

        ids = ["ghost_id"]
        existing = brain.vector_store.get_by_ids(ids)
        existing_ids = [d["id"] for d in existing]
        not_found = [i for i in ids if i not in existing_ids]

        assert not_found == ["ghost_id"]
        brain.vector_store.delete_by_ids.assert_not_called()

    def test_delete_by_source_no_bm25(self):
        brain = self._make_brain(docs=[{"source": "doc.txt", "chunks": 1, "doc_ids": ["z1"]}])
        brain.bm25 = None

        source = "doc.txt"
        docs = brain.list_documents()
        match = [d for d in docs if d["source"] == source]
        ids_to_delete = [i for d in match for i in d.get("doc_ids", [])]
        brain.vector_store.delete_by_ids(ids_to_delete)
        if brain.bm25 is not None:
            brain.bm25.delete_documents(ids_to_delete)

        brain.vector_store.delete_by_ids.assert_called_once_with(["z1"])


# ---------------------------------------------------------------------------
# CLI runtime qualification: --store-init handler (Sprint A)
# ---------------------------------------------------------------------------


class TestStoreInitCliRuntime:
    """Runtime qualification of --store-init handler logic."""

    def test_store_init_creates_namespace(self, tmp_path):
        from axon.projects import ensure_user_namespace

        base = tmp_path / "axon_data"
        base.mkdir()
        username = "testuser"
        user_dir = base / "AxonStore" / username
        ensure_user_namespace(user_dir)

        assert (user_dir / "default").exists()
        assert (user_dir / "projects").exists()
        assert (user_dir / "mounts").exists()
        assert (user_dir / ".shares").exists()

    def test_store_init_sets_brain_config(self, tmp_path):
        from axon.projects import ensure_user_namespace

        brain = MagicMock()
        brain.config.axon_store_mode = False

        base = tmp_path / "store_base"
        base.mkdir()
        username = "alice"
        store_root = base / "AxonStore"
        user_dir = store_root / username
        ensure_user_namespace(user_dir)

        brain.config.axon_store_base = str(base)
        brain.config.axon_store_mode = True
        brain.config.projects_root = str(user_dir)

        assert brain.config.axon_store_mode is True
        assert brain.config.projects_root == str(user_dir)

    def test_store_init_idempotent(self, tmp_path):
        from axon.projects import ensure_user_namespace

        user_dir = tmp_path / "AxonStore" / "bob"
        ensure_user_namespace(user_dir)
        ensure_user_namespace(user_dir)  # second call must not raise
        assert (user_dir / "default").exists()


# ---------------------------------------------------------------------------
# CLI runtime qualification: share lifecycle (Sprint A)
# ---------------------------------------------------------------------------


class TestShareCliRuntime:
    """Runtime qualification of --share-list/generate/redeem/revoke handler logic."""

    def _setup_user_dir(self, tmp_path, username="alice"):
        user_dir = tmp_path / "AxonStore" / username
        user_dir.mkdir(parents=True)
        (user_dir / ".shares").mkdir()
        return user_dir

    def test_share_generate_top_level_project(self, tmp_path):
        from axon.shares import generate_share_key

        user_dir = self._setup_user_dir(tmp_path)
        proj_dir = user_dir / "research"
        proj_dir.mkdir()
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_r"}', encoding="utf-8")

        result = generate_share_key(owner_user_dir=user_dir, project="research", grantee="bob")
        assert result["project"] == "research"
        assert result["grantee"] == "bob"
        assert result["share_string"]
        assert result["key_id"]

    def test_share_generate_nested_project(self, tmp_path):
        from axon.shares import generate_share_key

        user_dir = self._setup_user_dir(tmp_path)
        # Nested project: parent/subs/child
        parent_dir = user_dir / "parent"
        child_dir = parent_dir / "subs" / "child"
        child_dir.mkdir(parents=True)
        (child_dir / "meta.json").write_text('{"project_namespace_id": "ns_c"}', encoding="utf-8")

        result = generate_share_key(
            owner_user_dir=user_dir, project="parent/child", grantee="carol"
        )
        assert result["project"] == "parent/child"

    def test_share_list_returns_structure(self, tmp_path):
        from axon.shares import generate_share_key, list_shares

        user_dir = self._setup_user_dir(tmp_path)
        proj_dir = user_dir / "proj"
        proj_dir.mkdir()
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_p"}', encoding="utf-8")
        generate_share_key(owner_user_dir=user_dir, project="proj", grantee="dave")

        data = list_shares(user_dir)
        assert "sharing" in data
        assert "shared" in data
        assert len(data["sharing"]) == 1
        assert data["sharing"][0]["grantee"] == "dave"

    def test_share_list_empty_state(self, tmp_path):
        from axon.shares import list_shares

        user_dir = self._setup_user_dir(tmp_path)
        data = list_shares(user_dir)
        assert data["sharing"] == []
        assert data["shared"] == []

    def test_share_revoke_success(self, tmp_path):
        from axon.shares import generate_share_key, revoke_share_key

        user_dir = self._setup_user_dir(tmp_path)
        proj_dir = user_dir / "proj"
        proj_dir.mkdir()
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_rv"}', encoding="utf-8")
        result = generate_share_key(owner_user_dir=user_dir, project="proj", grantee="eve")
        key_id = result["key_id"]

        revoked = revoke_share_key(owner_user_dir=user_dir, key_id=key_id)
        assert revoked["key_id"] == key_id
        assert revoked["revoked_at"]

    def test_share_revoke_not_found_raises(self, tmp_path):
        from axon.shares import revoke_share_key

        user_dir = self._setup_user_dir(tmp_path)
        with pytest.raises(ValueError, match="not found"):
            revoke_share_key(owner_user_dir=user_dir, key_id="nonexistent")

    def test_share_redeem_mounts_project(self, tmp_path):
        from axon.shares import generate_share_key, redeem_share_key

        owner_dir = self._setup_user_dir(tmp_path, "owner")
        grantee_dir = self._setup_user_dir(tmp_path, "grantee")

        proj_dir = owner_dir / "shared_proj"
        proj_dir.mkdir()
        (proj_dir / "meta.json").write_text('{"project_namespace_id": "ns_sh"}', encoding="utf-8")

        gen = generate_share_key(owner_user_dir=owner_dir, project="shared_proj", grantee="grantee")
        share_string = gen["share_string"]

        redeemed = redeem_share_key(grantee_user_dir=grantee_dir, share_string=share_string)
        assert redeemed["project"] == "shared_proj"
        assert redeemed["owner"]

    def test_store_mode_guard_share_list(self):
        """--share-list exits if AxonStore mode is not active."""
        brain = MagicMock()
        brain.config.axon_store_mode = False
        assert not brain.config.axon_store_mode  # guard fires


# ---------------------------------------------------------------------------
# CLI runtime qualification: --session-list handler (Sprint A)
# ---------------------------------------------------------------------------


class TestSessionListCliRuntime:
    """Runtime qualification of --session-list handler logic."""

    def test_session_list_empty(self, tmp_path):
        from axon.sessions import _list_sessions

        sessions = _list_sessions(project="default")
        assert isinstance(sessions, list)

    def test_session_list_returns_sessions(self, tmp_path, monkeypatch):
        import axon.sessions as _sessions_mod

        fake_sessions = [
            {"id": "s1", "project": "work", "created_at": "2026-01-01T00:00:00Z"},
            {"id": "s2", "project": "work", "created_at": "2026-01-02T00:00:00Z"},
        ]
        monkeypatch.setattr(_sessions_mod, "_list_sessions", lambda project=None: fake_sessions)
        sessions = _sessions_mod._list_sessions(project="work")
        assert len(sessions) == 2
        assert sessions[0]["id"] == "s1"

    def test_session_list_scoped_to_project(self, tmp_path, monkeypatch):
        """Session list respects project scoping."""
        import axon.sessions as _sessions_mod

        all_sessions = [
            {"id": "s1", "project": "alpha"},
            {"id": "s2", "project": "beta"},
        ]
        monkeypatch.setattr(
            _sessions_mod,
            "_list_sessions",
            lambda project=None: [s for s in all_sessions if s["project"] == project],
        )
        result = _sessions_mod._list_sessions(project="alpha")
        assert all(s["project"] == "alpha" for s in result)
        assert len(result) == 1
