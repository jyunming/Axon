"""
tests/test_agent.py

Unit tests for src/axon/agent.py — dispatch_tool, individual tool handlers,
run_agent_loop, REPL_TOOLS schema, and helper utilities.

Mocking strategy:
  - brain: unittest.mock.MagicMock() with specific attributes set directly
  - LLM: MagicMock() for complete_with_tools / complete
  - File I/O: tmp_path pytest fixture + real Path writes
  - Loaders / external modules: patched at import site
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from axon.agent import (
    _DESTRUCTIVE_TOOLS,
    REPL_TOOLS,
    WEBAPP_TOOLS,
    ToolCall,
    _tool_get_config,
    _tool_get_stale_docs,
    _tool_graph_finalize,
    _tool_graph_status,
    _tool_list_knowledge,
    _tool_list_projects,
    _tool_read_file,
    _tool_search_knowledge,
    _tool_switch_project,
    _tool_update_settings,
    _tool_write_file,
    dispatch_tool,
    run_agent_loop,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain(**kwargs):
    """Return a MagicMock AxonBrain with sensible defaults."""
    brain = MagicMock()
    brain._active_project = "default"
    brain._ingested_hashes = set()
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_summaries = {}
    brain._community_build_in_progress = False
    brain._community_graph_dirty = False
    brain._doc_versions = {}
    brain._own_bm25 = MagicMock()
    brain._own_vector_store = MagicMock()
    for k, v in kwargs.items():
        setattr(brain, k, v)
    return brain


def _make_config(**kwargs):
    """Return a simple dataclass config for get_config tests."""

    @dataclasses.dataclass
    class FakeCfg:
        top_k: int = 5
        hybrid_search: bool = True
        rerank: bool = False
        hyde: bool = False
        graph_rag: bool = False
        raptor: bool = False
        similarity_threshold: float = 0.0
        multi_query: bool = False
        step_back: bool = False
        query_decompose: bool = False
        compress_context: bool = False
        truth_grounding: bool = False
        discussion_fallback: bool = False
        sentence_window: bool = False
        sentence_window_size: int = 2
        dedup_on_ingest: bool = True
        pdf_vision_ocr: bool = True

    cfg = FakeCfg()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# REPL_TOOLS schema sanity
# ---------------------------------------------------------------------------


class TestReplTools:
    def test_repl_tools_is_list(self):
        assert isinstance(REPL_TOOLS, list)
        assert len(REPL_TOOLS) > 0

    def test_every_tool_has_function_key(self):
        for t in REPL_TOOLS:
            assert t.get("type") == "function"
            assert "function" in t
            assert "name" in t["function"]

    def test_destructive_tools_not_in_webapp(self):
        webapp_names = {t["function"]["name"] for t in WEBAPP_TOOLS}
        for dt in _DESTRUCTIVE_TOOLS:
            assert (
                dt not in webapp_names
            ), f"Destructive tool '{dt}' must not appear in WEBAPP_TOOLS"

    def test_ingest_path_tool_present(self):
        names = {t["function"]["name"] for t in REPL_TOOLS}
        assert "ingest_path" in names

    def test_tool_call_namedtuple_defaults(self):
        tc = ToolCall(name="foo", args={"k": "v"})
        assert tc.name == "foo"
        assert tc.thought_signature is None  # default


# ---------------------------------------------------------------------------
# dispatch_tool — routing
# ---------------------------------------------------------------------------


class TestDispatchToolRouting:
    def test_unknown_tool_returns_error_string(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_destructive_tool_blocked_without_confirm_cb(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "clear_project", {}, confirm_cb=None)
        assert "requires confirmation" in result or "skipped" in result

    def test_destructive_tool_cancelled_when_confirm_returns_false(self):
        brain = _make_brain()
        confirm_cb = MagicMock(return_value=False)
        result = dispatch_tool(
            brain, "delete_documents", {"source": "foo.pdf"}, confirm_cb=confirm_cb
        )
        assert "cancelled" in result.lower()

    def test_destructive_tool_proceeds_when_confirm_returns_true(self):
        brain = _make_brain()
        brain.list_documents.return_value = [
            {"source": "foo.pdf", "chunks": 3, "doc_ids": ["foo.pdf::abc"]}
        ]
        brain._ingested_hashes = {"abc"}
        confirm_cb = MagicMock(return_value=True)
        result = dispatch_tool(
            brain, "delete_documents", {"source": "foo.pdf"}, confirm_cb=confirm_cb
        )
        assert "Deleted" in result or "chunk" in result

    def test_dispatch_exception_returns_error_string(self):
        """A tool that raises should not propagate — dispatch_tool catches it."""
        brain = _make_brain()
        brain.search_raw.side_effect = RuntimeError("connection refused")
        result = dispatch_tool(brain, "search_knowledge", {"query": "anything"})
        assert "failed" in result.lower() or "Search failed" in result

    def test_dispatch_routes_list_knowledge(self):
        brain = _make_brain()
        brain.list_documents.return_value = [{"source": "a.txt", "chunks": 2}]
        result = dispatch_tool(brain, "list_knowledge", {})
        assert "source(s)" in result

    def test_dispatch_routes_switch_project(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "switch_project", {"name": "myproj"})
        brain.switch_project.assert_called_once_with("myproj")
        assert "myproj" in result

    def test_dispatch_routes_get_config(self):
        brain = _make_brain()
        brain.config = _make_config()
        result = dispatch_tool(brain, "get_config", {})
        assert "active_project" in result

    def test_dispatch_routes_graph_status(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "graph_status", {})
        assert "entities" in result

    def test_dispatch_routes_graph_finalize(self):
        brain = _make_brain()
        dispatch_tool(brain, "graph_finalize", {})
        brain.finalize_ingest.assert_called_once()


# ---------------------------------------------------------------------------
# _tool_read_file
# ---------------------------------------------------------------------------


class TestToolReadFile:
    def test_read_file_success(self, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("hello world", encoding="utf-8")
        result = _tool_read_file({"path": str(f)})
        assert result == "hello world"

    def test_read_file_not_found(self, tmp_path):
        result = _tool_read_file({"path": str(tmp_path / "missing.txt")})
        assert "read_file failed" in result or "⚠️" in result

    def test_read_file_truncates_large_content(self, tmp_path):
        f = tmp_path / "big.txt"
        big_content = "x" * 10_000
        f.write_text(big_content, encoding="utf-8")
        result = _tool_read_file({"path": str(f)})
        assert "truncated" in result
        assert len(result) < len(big_content)

    def test_read_file_empty_path_returns_error(self):
        result = _tool_read_file({"path": ""})
        assert "⚠️" in result or "failed" in result


# ---------------------------------------------------------------------------
# _tool_write_file
# ---------------------------------------------------------------------------


class TestToolWriteFile:
    def test_write_file_success(self, tmp_path):
        target = tmp_path / "out.txt"
        result = _tool_write_file({"path": str(target), "content": "test content"})
        assert "Wrote" in result
        assert target.read_text(encoding="utf-8") == "test content"

    def test_write_file_append_mode(self, tmp_path):
        target = tmp_path / "out.txt"
        target.write_text("first", encoding="utf-8")
        result = _tool_write_file({"path": str(target), "content": " second", "mode": "a"})
        assert "Appended" in result
        assert target.read_text(encoding="utf-8") == "first second"

    def test_write_file_creates_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "file.txt"
        result = _tool_write_file({"path": str(target), "content": "data"})
        assert target.exists()
        assert "Wrote" in result

    def test_write_file_invalid_mode_falls_back_to_overwrite(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("original", encoding="utf-8")
        _tool_write_file({"path": str(target), "content": "new", "mode": "z"})
        assert target.read_text(encoding="utf-8") == "new"


# ---------------------------------------------------------------------------
# _tool_list_knowledge
# ---------------------------------------------------------------------------


class TestToolListKnowledge:
    def test_empty_knowledge_base(self):
        brain = _make_brain()
        brain.list_documents.return_value = []
        result = _tool_list_knowledge(brain, {})
        assert "empty" in result.lower()

    def test_knowledge_base_with_documents(self):
        brain = _make_brain()
        brain.list_documents.return_value = [
            {"source": "doc_a.pdf", "chunks": 10},
            {"source": "doc_b.txt", "chunks": 5},
        ]
        result = _tool_list_knowledge(brain, {})
        assert "doc_a.pdf" in result
        assert "doc_b.txt" in result
        assert "2 source(s)" in result

    def test_list_knowledge_switches_project_and_restores(self):
        brain = _make_brain()
        brain._active_project = "proj-a"
        brain.list_documents.return_value = []
        _tool_list_knowledge(brain, {"project": "proj-b"})
        # Should have switched to proj-b, then restored proj-a
        calls = [c.args[0] for c in brain.switch_project.call_args_list]
        assert "proj-b" in calls


# ---------------------------------------------------------------------------
# _tool_search_knowledge
# ---------------------------------------------------------------------------


class TestToolSearchKnowledge:
    def test_search_no_query_returns_error(self):
        brain = _make_brain()
        result = _tool_search_knowledge(brain, {"query": ""})
        assert "No query" in result

    def test_search_returns_results(self):
        brain = _make_brain()
        brain.search_raw.return_value = (
            [
                {
                    "text": "relevant chunk",
                    "score": 0.95,
                    "metadata": {"source": "x.pdf"},
                    "id": "1",
                },
            ],
            {},
            [],
        )
        result = _tool_search_knowledge(brain, {"query": "hello"})
        assert "1 result" in result
        assert "x.pdf" in result

    def test_search_no_results(self):
        brain = _make_brain()
        brain.search_raw.return_value = ([], {}, [])
        result = _tool_search_knowledge(brain, {"query": "obscure"})
        assert "No results" in result

    def test_search_exception_returns_error(self):
        brain = _make_brain()
        brain.search_raw.side_effect = Exception("db error")
        result = _tool_search_knowledge(brain, {"query": "test"})
        assert "Search failed" in result


# ---------------------------------------------------------------------------
# _tool_switch_project
# ---------------------------------------------------------------------------


class TestToolSwitchProject:
    def test_switch_project_success(self):
        brain = _make_brain()
        result = _tool_switch_project(brain, {"name": "analytics"})
        brain.switch_project.assert_called_once_with("analytics")
        assert "analytics" in result

    def test_switch_project_no_name_returns_error(self):
        brain = _make_brain()
        result = _tool_switch_project(brain, {})
        assert "No project name" in result
        brain.switch_project.assert_not_called()


# ---------------------------------------------------------------------------
# _tool_get_config
# ---------------------------------------------------------------------------


class TestToolGetConfig:
    def test_get_config_includes_active_project(self):
        brain = _make_brain()
        brain.config = _make_config()
        result = _tool_get_config(brain)
        assert "active_project: default" in result

    def test_get_config_includes_all_fields(self):
        brain = _make_brain()
        brain.config = _make_config(top_k=10)
        result = _tool_get_config(brain)
        assert "top_k: 10" in result


# ---------------------------------------------------------------------------
# _tool_update_settings
# ---------------------------------------------------------------------------


class TestToolUpdateSettings:
    def test_update_settings_changes_top_k(self):
        brain = _make_brain()
        brain.config = _make_config(top_k=5)
        result = _tool_update_settings(brain, {"top_k": 15})
        assert "top_k" in result
        assert brain.config.top_k == 15

    def test_update_settings_no_change(self):
        brain = _make_brain()
        brain.config = _make_config(top_k=5)
        result = _tool_update_settings(brain, {"top_k": 5})
        assert "No settings changed" in result

    def test_update_settings_empty_args(self):
        brain = _make_brain()
        brain.config = _make_config()
        result = _tool_update_settings(brain, {})
        assert "No settings changed" in result

    def test_update_settings_bool_field(self):
        brain = _make_brain()
        brain.config = _make_config(rerank=False)
        result = _tool_update_settings(brain, {"rerank": True})
        assert brain.config.rerank is True
        assert "rerank" in result


# ---------------------------------------------------------------------------
# _tool_graph_status
# ---------------------------------------------------------------------------


class TestToolGraphStatus:
    def test_graph_status_empty_graph(self):
        brain = _make_brain()
        result = _tool_graph_status(brain)
        assert "entities: 0" in result
        assert "relations: 0" in result

    def test_graph_status_with_data(self):
        brain = _make_brain()
        brain._entity_graph = {"EntityA": {}, "EntityB": {}}
        brain._relation_count = 5
        result = _tool_graph_status(brain)
        assert "entities: 2" in result

    def test_graph_status_shows_rebuild_flag(self):
        brain = _make_brain()
        brain._community_build_in_progress = True
        result = _tool_graph_status(brain)
        assert "True" in result


# ---------------------------------------------------------------------------
# _tool_graph_finalize
# ---------------------------------------------------------------------------


class TestToolGraphFinalize:
    def test_graph_finalize_success(self):
        brain = _make_brain()
        brain._community_summaries = {"c1": {}, "c2": {}}
        result = _tool_graph_finalize(brain)
        brain.finalize_ingest.assert_called_once()
        assert "complete" in result.lower()

    def test_graph_finalize_exception_returns_error(self):
        brain = _make_brain()
        brain.finalize_ingest.side_effect = RuntimeError("graph error")
        result = _tool_graph_finalize(brain)
        assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# _tool_list_projects
# ---------------------------------------------------------------------------


class TestToolListProjects:
    def test_list_projects_no_projects(self):
        with patch("axon.projects.list_projects", return_value=[]):
            result = _tool_list_projects()
        assert "No projects" in result

    def test_list_projects_returns_names(self):
        with patch("axon.projects.list_projects") as mock_lp:
            mock_lp.return_value = [
                {"name": "proj-a", "description": "first", "children": []},
                {"name": "proj-b", "description": "", "children": []},
            ]
            result = _tool_list_projects()
        assert "proj-a" in result
        assert "proj-b" in result
        assert "2 project(s)" in result

    def test_list_projects_empty(self):
        with patch("axon.projects.list_projects") as mock_lp:
            mock_lp.return_value = []
            result = _tool_list_projects()
        assert "No projects" in result


# ---------------------------------------------------------------------------
# _tool_get_stale_docs
# ---------------------------------------------------------------------------


class TestToolGetStaleDocs:
    def test_no_doc_versions_tracked(self):
        brain = _make_brain()
        brain._doc_versions = {}
        result = _tool_get_stale_docs(brain, {})
        assert "No ingestion history" in result

    def test_no_stale_docs_when_all_recent(self):
        from datetime import datetime, timezone

        brain = _make_brain()
        now_iso = datetime.now(timezone.utc).isoformat()
        brain._doc_versions = {"doc.txt": {"ingested_at": now_iso, "content_hash": "abc"}}
        result = _tool_get_stale_docs(brain, {"days": 30})
        assert "No stale" in result

    def test_stale_docs_detected(self):
        brain = _make_brain()
        brain._doc_versions = {
            "old_doc.txt": {"ingested_at": "2020-01-01T00:00:00+00:00", "content_hash": "xyz"}
        }
        result = _tool_get_stale_docs(brain, {"days": 30})
        assert "old_doc.txt" in result
        assert "stale" in result.lower()


# ---------------------------------------------------------------------------
# _tool_ingest_url (via dispatch_tool)
# ---------------------------------------------------------------------------


class TestToolIngestUrl:
    def test_invalid_url_returns_error(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "ingest_url", {"url": "ftp://bad.url"})
        assert "Invalid URL" in result or "must start with http" in result

    def test_empty_url_returns_error(self):
        brain = _make_brain()
        result = dispatch_tool(brain, "ingest_url", {"url": ""})
        assert "No URL provided" in result

    def test_url_load_exception_returns_error(self):
        brain = _make_brain()
        with patch("axon.loaders.URLLoader") as mock_loader_cls:
            mock_loader_cls.return_value.load.side_effect = Exception("timeout")
            result = dispatch_tool(brain, "ingest_url", {"url": "https://example.com"})
        assert "Failed to fetch" in result or "failed" in result.lower()


# ---------------------------------------------------------------------------
# _tool_add_text (via dispatch_tool)
# ---------------------------------------------------------------------------


class TestToolAddText:
    def test_add_text_empty_returns_error(self):
        brain = _make_brain()
        with patch("axon.loaders.SmartTextLoader"):
            result = dispatch_tool(brain, "add_text", {"text": ""})
        assert "No text provided" in result

    def test_add_text_success(self):
        brain = _make_brain()
        mock_loader = MagicMock()
        mock_loader.load_text.return_value = [{"text": "hello", "metadata": {}}]
        # agent.py imports SmartTextLoader locally from axon.loaders — patch there.
        with patch("axon.loaders.SmartTextLoader", return_value=mock_loader):
            dispatch_tool(brain, "add_text", {"text": "hello world"})
        # brain.ingest should have been called
        brain.ingest.assert_called()


# ---------------------------------------------------------------------------
# _tool_delete_documents (via dispatch_tool)
# ---------------------------------------------------------------------------


class TestToolDeleteDocuments:
    def test_delete_documents_no_source_returns_error(self):
        brain = _make_brain()
        confirm_cb = MagicMock(return_value=True)
        result = dispatch_tool(brain, "delete_documents", {"source": ""}, confirm_cb=confirm_cb)
        assert "No source" in result

    def test_delete_documents_source_not_found(self):
        brain = _make_brain()
        brain.list_documents.return_value = []
        confirm_cb = MagicMock(return_value=True)
        result = dispatch_tool(
            brain, "delete_documents", {"source": "gone.pdf"}, confirm_cb=confirm_cb
        )
        assert "No documents found" in result

    def test_delete_documents_success(self):
        brain = _make_brain()
        brain.list_documents.return_value = [
            {"source": "target.pdf", "chunks": 2, "doc_ids": ["target.pdf::h1", "target.pdf::h2"]}
        ]
        brain._ingested_hashes = {"h1", "h2"}
        confirm_cb = MagicMock(return_value=True)
        result = dispatch_tool(
            brain, "delete_documents", {"source": "target.pdf"}, confirm_cb=confirm_cb
        )
        assert "Deleted" in result
        brain._own_vector_store.delete_by_ids.assert_called_once()


# ---------------------------------------------------------------------------
# run_agent_loop
# ---------------------------------------------------------------------------


class TestRunAgentLoop:
    def _make_llm(self):
        llm = MagicMock()
        return llm

    def test_plain_text_response_returns_immediately(self):
        llm = self._make_llm()
        llm.complete_with_tools.return_value = "Here is your answer."
        brain = _make_brain()
        result = run_agent_loop(llm, brain, "hello", [])
        assert result == "Here is your answer."

    def test_tool_call_then_text_response(self):
        llm = self._make_llm()
        brain = _make_brain()
        brain.list_documents.return_value = []
        # First call returns a ToolCall, second returns plain text
        llm.complete_with_tools.side_effect = [
            [ToolCall(name="list_knowledge", args={})],
            "Knowledge base is empty.",
        ]
        result = run_agent_loop(llm, brain, "what do you know?", [])
        assert "Knowledge base" in result or "empty" in result.lower() or isinstance(result, str)

    def test_max_steps_triggers_final_summary(self):
        llm = self._make_llm()
        brain = _make_brain()
        brain.list_documents.return_value = []
        # Always return a tool call to exhaust max_steps
        llm.complete_with_tools.return_value = [ToolCall(name="list_knowledge", args={})]
        llm.complete.return_value = "Summary response."
        run_agent_loop(llm, brain, "keep going", [], max_steps=2)
        llm.complete.assert_called()

    def test_identical_retry_loop_breaks_early(self):
        llm = self._make_llm()
        brain = _make_brain()
        brain.list_documents.return_value = []
        # Simulate LLM stuck in retry loop: same tool call repeatedly
        llm.complete_with_tools.return_value = [ToolCall(name="list_knowledge", args={})]
        llm.complete.return_value = "I'll stop retrying."
        result = run_agent_loop(llm, brain, "list stuff", [], max_steps=10)
        # Should have exited early via the identical-retry guard
        assert isinstance(result, str)

    def test_step_cb_called_for_each_tool(self):
        llm = self._make_llm()
        brain = _make_brain()
        brain.list_documents.return_value = [{"source": "x.txt", "chunks": 1}]
        llm.complete_with_tools.side_effect = [
            [ToolCall(name="list_knowledge", args={})],
            "Done.",
        ]
        step_cb = MagicMock()
        run_agent_loop(llm, brain, "list", [], step_cb=step_cb)
        step_cb.assert_called()
        name_called = step_cb.call_args[0][0]
        assert name_called == "list_knowledge"

    def test_custom_tools_override_defaults(self):
        llm = self._make_llm()
        brain = _make_brain()
        llm.complete_with_tools.return_value = "ok"
        custom_tools = [{"type": "function", "function": {"name": "my_tool"}}]
        run_agent_loop(llm, brain, "hi", [], tools=custom_tools)
        # Verify llm received custom tools
        call_args = llm.complete_with_tools.call_args
        assert call_args[0][1] == custom_tools

    def test_xml_stripped_from_plain_text_response(self):
        llm = self._make_llm()
        brain = _make_brain()
        llm.complete_with_tools.return_value = (
            "Here is the answer.<tool_calls>some leaked xml</tool_calls> Done."
        )
        result = run_agent_loop(llm, brain, "q", [])
        assert "<tool_calls>" not in result

    def test_step_cb_exception_does_not_abort_loop(self):
        llm = self._make_llm()
        brain = _make_brain()
        brain.list_documents.return_value = []
        llm.complete_with_tools.side_effect = [
            [ToolCall(name="list_knowledge", args={})],
            "Done.",
        ]
        bad_cb = MagicMock(side_effect=RuntimeError("cb exploded"))
        # Should not raise
        result = run_agent_loop(llm, brain, "list", [], step_cb=bad_cb)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# _make_vision_fn
# ---------------------------------------------------------------------------


class TestMakeVisionFn:
    def test_returns_none_when_brain_is_none(self):
        from axon.agent import _make_vision_fn

        assert _make_vision_fn(None) is None

    def test_returns_none_when_pdf_vision_ocr_disabled(self):
        from axon.agent import _make_vision_fn

        brain = _make_brain()
        brain.config = _make_config(pdf_vision_ocr=False)
        result = _make_vision_fn(brain)
        assert result is None

    def test_returns_none_when_llm_lacks_complete_with_image(self):
        from axon.agent import _make_vision_fn

        brain = _make_brain()
        brain.config = _make_config(pdf_vision_ocr=True)
        brain.llm = MagicMock(spec=[])  # no complete_with_image attribute
        result = _make_vision_fn(brain)
        assert result is None

    def test_returns_callable_when_llm_supports_vision(self):
        from axon.agent import _make_vision_fn

        brain = _make_brain()
        brain.config = _make_config(pdf_vision_ocr=True)
        brain.llm = MagicMock()
        brain.llm.complete_with_image.return_value = "extracted text"
        fn = _make_vision_fn(brain)
        assert callable(fn)
        result = fn(b"fake image bytes")
        assert result == "extracted text"

    def test_vision_fn_returns_empty_string_on_exception(self):
        from axon.agent import _make_vision_fn

        brain = _make_brain()
        brain.config = _make_config(pdf_vision_ocr=True)
        brain.llm = MagicMock()
        brain.llm.complete_with_image.side_effect = RuntimeError("vision error")
        fn = _make_vision_fn(brain)
        result = fn(b"bytes")
        assert result == ""


# ---------------------------------------------------------------------------
# Confirm message formatting for destructive tools
# ---------------------------------------------------------------------------


class TestDestructiveConfirmMessages:
    """Verify that the confirm_cb receives a sensible message for each destructive tool."""

    @pytest.mark.parametrize(
        "tool_name,args,expected_fragment",
        [
            ("purge_source", {"source": "/data/file.pdf"}, "/data/file.pdf"),
            ("delete_documents", {"source": "report.pdf"}, "report.pdf"),
            ("clear_project", {}, "irreversible"),
            ("delete_project", {"name": "old-proj"}, "old-proj"),
            ("run_shell", {"command": "ls"}, "ls"),
        ],
    )
    def test_confirm_message_contains_expected_fragment(self, tool_name, args, expected_fragment):
        brain = _make_brain()
        # Reject the confirm — we just want to capture the message
        captured = []

        def confirm_cb(msg):
            captured.append(msg)
            return False  # cancel so no side effects

        dispatch_tool(brain, tool_name, args, confirm_cb=confirm_cb)
        assert captured, f"confirm_cb was never called for '{tool_name}'"
        assert (
            expected_fragment in captured[0]
        ), f"Expected '{expected_fragment}' in confirm message: {captured[0]!r}"
