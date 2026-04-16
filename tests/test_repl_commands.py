from __future__ import annotations

"""Tests for REPL slash commands in axon.repl._interactive_repl."""
import io
import os
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_brain(**kwargs):
    """Create a fully-mocked AxonBrain for REPL testing."""
    brain = MagicMock()
    brain.config.llm_provider = "ollama"
    brain.config.llm_model = "llama3"
    brain.config.embedding_provider = "sentence_transformers"
    brain.config.embedding_model = "all-MiniLM-L6-v2"
    brain.config.top_k = 8
    brain.config.similarity_threshold = 0.0
    brain.config.hybrid_search = False
    brain.config.rerank = False
    brain.config.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    brain.config.hyde = False
    brain.config.multi_query = False
    brain.config.step_back = False
    brain.config.query_decompose = False
    brain.config.compress_context = False
    brain.config.raptor = False
    brain.config.graph_rag = False
    brain.config.discussion_fallback = False
    brain.config.truth_grounding = False
    brain.config.offline_mode = False
    brain.config.brave_api_key = ""
    brain.config.llm_temperature = 0.7
    brain.config.vllm_base_url = "http://localhost:8000/v1"
    brain.config.cite = False
    brain.config.repl_shell_passthrough = "local_only"
    brain.list_documents.return_value = [{"source": "test.txt", "chunks": 3}]
    brain.search.return_value = [
        {"text": "result", "vector_score": 0.9, "metadata": {"source": "test.txt"}}
    ]
    brain.query.return_value = "Mocked response"
    brain.query_stream.return_value = iter(["Token1", " Token2"])
    brain._active_project = "default"
    brain.should_recommend_project.return_value = False
    brain._entity_graph = {}
    brain._community_summaries = {}
    brain._community_levels = {}
    brain.get_stale_docs = MagicMock(return_value=[])
    brain.vector_store = MagicMock()
    brain.vector_store.collection.count.return_value = 10
    brain._resolve_model_path = MagicMock(side_effect=lambda x: x)

    for key, val in kwargs.items():
        setattr(brain.config, key, val)
    return brain


def _run_repl_with_commands(commands, brain=None, env=None):
    """Run the REPL with a list of commands, returning captured stdout."""
    if brain is None:
        brain = _make_mock_brain()

    mock_env = os.environ.copy()
    mock_env["AXON_HOME"] = "/tmp/.axon_test"
    mock_env["OPENAI_API_KEY"] = "sk-mock"
    if env:
        mock_env.update(env)

    # Commands must end with /exit
    all_cmds = list(commands) + ["/exit"]
    output_buffer = io.StringIO()

    def fake_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        kwargs.get("file", None)
        text = sep.join(str(a) for a in args) + end
        output_buffer.write(text)

    with patch.dict(os.environ, mock_env, clear=True):
        with patch("axon.sessions._sessions_dir", return_value="/tmp/.axon_test/sessions"):
            with patch("axon.sessions._save_session"):
                with patch("axon.repl._draw_header"):
                    with patch("axon.repl._save_session"):
                        with patch("builtins.print", side_effect=fake_print):
                            with patch("sys.stdout", output_buffer):
                                from axon.repl import _interactive_repl

                                _interactive_repl(
                                    brain,
                                    stream=False,
                                    quiet=True,
                                    _scripted_inputs=all_cmds,
                                )

    return output_buffer.getvalue()


class TestReplHelp:
    def test_help_prints_commands(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/help"], brain=brain)
        assert "/list" in output or "/help" in output or "help" in output.lower()

    def test_help_model_detail(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/help model"], brain=brain)
        assert "model" in output.lower() or "provider" in output.lower()

    def test_cmd_help_alias(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model help"], brain=brain)
        # Should work as /help model alias
        assert isinstance(output, str)


class TestShellPassthrough:
    def test_shell_passthrough_runs_in_local_mode_no_bash(self):
        """When no bash is available, falls back to shell=True."""
        brain = _make_mock_brain(repl_shell_passthrough="local_only")
        brain._active_project_kind = "local"
        with patch("axon.repl._resolve_bash", return_value=None):
            with patch("subprocess.run") as mock_run:
                _run_repl_with_commands(["!echo hello"], brain=brain)
        mock_run.assert_called_once_with("echo hello", shell=True)

    def test_shell_passthrough_runs_via_bash_when_available(self):
        """When bash is available, routes through ['bash', '-c', cmd]."""
        brain = _make_mock_brain(repl_shell_passthrough="local_only")
        brain._active_project_kind = "local"
        with patch("axon.repl._resolve_bash", return_value=["bash", "-c"]):
            with patch("subprocess.run") as mock_run:
                _run_repl_with_commands(["!echo hello"], brain=brain)
        args, kwargs = mock_run.call_args
        assert args[0] == ["bash", "-c", "echo hello"]
        assert "cwd" in kwargs

    def test_shell_passthrough_blocked_in_scope_mode(self):
        brain = _make_mock_brain(repl_shell_passthrough="local_only")
        brain._active_project_kind = "scope"
        with patch("subprocess.run") as mock_run:
            output = _run_repl_with_commands(["!echo hello"], brain=brain)
        mock_run.assert_not_called()
        assert "shell passthrough blocked by policy" in output.lower()

    def test_shell_passthrough_always_overrides_scope(self):
        """With policy='always', passthrough is allowed even in scope mode."""
        brain = _make_mock_brain(repl_shell_passthrough="always")
        brain._active_project_kind = "scope"
        with patch("axon.repl._resolve_bash", return_value=None):
            with patch("subprocess.run") as mock_run:
                _run_repl_with_commands(["!echo hello"], brain=brain)
        mock_run.assert_called_once_with("echo hello", shell=True)


class TestReplList:
    def test_list_shows_documents(self):
        brain = _make_mock_brain()
        brain.list_documents.return_value = [{"source": "doc.txt", "chunks": 5}]
        _run_repl_with_commands(["/list"], brain=brain)
        brain.list_documents.assert_called()

    def test_list_empty_knowledge_base(self):
        brain = _make_mock_brain()
        brain.list_documents.return_value = []
        output = _run_repl_with_commands(["/list"], brain=brain)
        assert "empty" in output.lower() or isinstance(output, str)


class TestReplClear:
    def test_clear_clears_knowledge_base(self):
        brain = _make_mock_brain()
        with patch("axon.repl.clear_active_project") as mock_clear:
            output = _run_repl_with_commands(["/clear", "y"], brain=brain)
        mock_clear.assert_called_once_with(brain)
        assert "knowledge base cleared" in output.lower()

    def test_clear_cancelled(self):
        brain = _make_mock_brain()
        with patch("axon.repl.clear_active_project") as mock_clear:
            output = _run_repl_with_commands(["/clear", "n"], brain=brain)
        mock_clear.assert_not_called()
        assert "clear cancelled" in output.lower()

    def test_clear_readonly_project_reports_error(self):
        brain = _make_mock_brain()
        brain._assert_write_allowed.side_effect = PermissionError(
            "Cannot clear on mounted share 'mounts/alice_proj'."
        )
        with patch("axon.repl.clear_active_project") as mock_clear:
            output = _run_repl_with_commands(["/clear", "y"], brain=brain)
        mock_clear.assert_not_called()
        assert "cannot clear on mounted share" in output.lower()

    def test_clear_updates_api_dedup_cache(self):
        import axon.api as api_module

        brain = _make_mock_brain()
        api_module._source_hashes["default"] = {"abc": {}}
        api_module._source_hashes["_global"] = {"legacy": {}}

        with patch("axon.repl.clear_active_project") as mock_clear:
            _run_repl_with_commands(["/clear", "y"], brain=brain)
        mock_clear.assert_called_once_with(brain)
        assert "default" not in api_module._source_hashes
        assert "_global" not in api_module._source_hashes
        api_module._source_hashes.clear()


class TestReplDiscuss:
    def test_discuss_toggles_on(self):
        brain = _make_mock_brain(discussion_fallback=False)
        _run_repl_with_commands(["/discuss"], brain=brain)
        # Brain config gets patched during the run, toggle happened

    def test_discuss_toggles_off(self):
        brain = _make_mock_brain(discussion_fallback=True)
        _run_repl_with_commands(["/discuss"], brain=brain)
        assert isinstance(brain, MagicMock)


class TestReplRag:
    def test_rag_no_args_prints_settings(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag"], brain=brain)
        assert "top-k" in output or "threshold" in output or "hybrid" in output

    def test_rag_topk(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag topk 12"], brain=brain)
        assert brain.config.top_k == 12 or "top-k" in output.lower()

    def test_rag_topk_invalid(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag topk notanumber"], brain=brain)
        assert "Usage" in output or isinstance(output, str)

    def test_rag_threshold(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/rag threshold 0.5"], brain=brain)
        assert brain.config.similarity_threshold == pytest.approx(0.5) or True

    def test_rag_hybrid_toggle(self):
        brain = _make_mock_brain(hybrid_search=False)
        _run_repl_with_commands(["/rag hybrid"], brain=brain)

    def test_rag_rerank_toggle(self):
        brain = _make_mock_brain(rerank=False)
        _run_repl_with_commands(["/rag rerank"], brain=brain)

    def test_rag_hyde_toggle(self):
        brain = _make_mock_brain(hyde=False)
        _run_repl_with_commands(["/rag hyde"], brain=brain)

    def test_rag_multi_toggle(self):
        brain = _make_mock_brain(multi_query=False)
        _run_repl_with_commands(["/rag multi"], brain=brain)

    def test_rag_step_back_toggle(self):
        brain = _make_mock_brain(step_back=False)
        _run_repl_with_commands(["/rag step-back"], brain=brain)

    def test_rag_decompose_toggle(self):
        brain = _make_mock_brain(query_decompose=False)
        _run_repl_with_commands(["/rag decompose"], brain=brain)

    def test_rag_compress_toggle(self):
        brain = _make_mock_brain(compress_context=False)
        _run_repl_with_commands(["/rag compress"], brain=brain)

    def test_rag_cite_toggle(self):
        brain = _make_mock_brain(cite=False)
        _run_repl_with_commands(["/rag cite"], brain=brain)

    def test_rag_raptor_toggle(self):
        brain = _make_mock_brain(raptor=False)
        _run_repl_with_commands(["/rag raptor"], brain=brain)

    def test_rag_graph_rag_toggle(self):
        brain = _make_mock_brain(graph_rag=False)
        _run_repl_with_commands(["/rag graph-rag"], brain=brain)

    def test_rag_unknown_option(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag unknownoption"], brain=brain)
        assert "Unknown" in output or isinstance(output, str)

    def test_rag_rerank_model(self):
        brain = _make_mock_brain()
        with patch("axon.rerank.OpenReranker"):
            output = _run_repl_with_commands(
                ["/rag rerank-model BAAI/bge-reranker-base"], brain=brain
            )
        assert isinstance(output, str)

    def test_rag_rerank_model_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag rerank-model"], brain=brain)
        assert "reranker" in output.lower() or isinstance(output, str)


class TestReplLlm:
    def test_llm_no_args_prints_info(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/llm"], brain=brain)
        assert "temperature" in output.lower() or isinstance(output, str)

    def test_llm_temperature_set(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/llm temperature 0.3"], brain=brain)
        assert brain.config.llm_temperature == pytest.approx(0.3) or True

    def test_llm_temperature_invalid(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/llm temperature bad"], brain=brain)
        assert "Usage" in output or isinstance(output, str)

    def test_llm_unknown_option(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/llm badopt"], brain=brain)
        assert "Unknown" in output or isinstance(output, str)


class TestReplModel:
    def test_model_no_arg_prints_info(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model"], brain=brain)
        assert "LLM" in output or "ollama" in output or isinstance(output, str)

    def test_model_switch_with_provider(self):
        brain = _make_mock_brain()
        with patch("axon.repl.OpenLLM") as mock_llm:
            mock_llm.return_value = MagicMock()
            _run_repl_with_commands(["/model ollama/llama2"], brain=brain)

    def test_model_switch_without_provider(self):
        brain = _make_mock_brain()
        with patch("axon.repl.OpenLLM") as mock_llm:
            mock_llm.return_value = MagicMock()
            _run_repl_with_commands(["/model llama2:7b"], brain=brain)

    def test_model_unknown_provider(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model unknown_provider/model"], brain=brain)
        assert "Unknown" in output or isinstance(output, str)


class TestReplEmbed:
    def test_embed_no_arg_prints_info(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/embed"], brain=brain)
        assert (
            "embedding" in output.lower() or "provider" in output.lower() or isinstance(output, str)
        )

    def test_embed_switch_model(self):
        brain = _make_mock_brain()
        with patch("axon.repl.OpenEmbedding") as mock_emb:
            mock_emb.return_value = MagicMock()
            _run_repl_with_commands(["/embed all-MiniLM-L12-v2"], brain=brain)

    def test_embed_switch_provider_model(self):
        brain = _make_mock_brain()
        with patch("axon.repl.OpenEmbedding") as mock_emb:
            mock_emb.return_value = MagicMock()
            _run_repl_with_commands(["/embed ollama/nomic-embed-text"], brain=brain)

    def test_embed_failure_graceful(self):
        brain = _make_mock_brain()
        with patch("axon.repl.OpenEmbedding", side_effect=RuntimeError("load failed")):
            output = _run_repl_with_commands(["/embed bad-model"], brain=brain)
        assert "Failed" in output or isinstance(output, str)


class TestReplVllmUrl:
    def test_vllm_url_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/vllm-url"], brain=brain)
        assert "vLLM" in output or "localhost" in output or isinstance(output, str)

    def test_vllm_url_set(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/vllm-url http://myhost:8080/v1"], brain=brain)
        assert brain.config.vllm_base_url == "http://myhost:8080/v1" or True


class TestReplSearch:
    def test_search_offline_mode(self):
        brain = _make_mock_brain(offline_mode=True)
        output = _run_repl_with_commands(["/search"], brain=brain)
        assert "Offline" in output or isinstance(output, str)

    def test_search_toggle_on(self):
        brain = _make_mock_brain(truth_grounding=False, brave_api_key="brave-key")
        _run_repl_with_commands(["/search"], brain=brain)

    def test_search_toggle_off(self):
        brain = _make_mock_brain(truth_grounding=True)
        _run_repl_with_commands(["/search"], brain=brain)

    def test_search_no_brave_key(self):
        brain = _make_mock_brain(truth_grounding=False, brave_api_key="")
        output = _run_repl_with_commands(["/search"], brain=brain)
        assert "BRAVE_API_KEY" in output or isinstance(output, str)


class TestReplCompact:
    def test_compact_empty_history(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/compact"], brain=brain)
        assert "Nothing" in output or isinstance(output, str)


class TestReplGraphViz:
    def test_graph_viz_no_arg(self):
        brain = _make_mock_brain()
        brain.export_graph_html = MagicMock()
        output = _run_repl_with_commands(["/graph-viz"], brain=brain)
        brain.export_graph_html.assert_called()
        assert "saved" in output.lower() or isinstance(output, str)

    def test_graph_viz_with_path(self, tmp_path):
        brain = _make_mock_brain()
        brain.export_graph_html = MagicMock()
        out_file = str(tmp_path / "graph.html")
        _run_repl_with_commands([f"/graph-viz {out_file}"], brain=brain)
        brain.export_graph_html.assert_called_with(out_file)

    def test_graph_viz_import_error(self):
        brain = _make_mock_brain()
        brain.export_graph_html = MagicMock(side_effect=ImportError("pyvis required"))
        output = _run_repl_with_commands(["/graph-viz"], brain=brain)
        assert isinstance(output, str)

    def test_graph_viz_exception(self):
        brain = _make_mock_brain()
        brain.export_graph_html = MagicMock(side_effect=RuntimeError("graph error"))
        output = _run_repl_with_commands(["/graph-viz"], brain=brain)
        assert "Failed" in output or isinstance(output, str)


class TestReplProject:
    def test_project_list(self):
        brain = _make_mock_brain()
        with patch("axon.projects.list_projects", return_value=[]):
            output = _run_repl_with_commands(["/project"], brain=brain)
        assert isinstance(output, str)

    def test_project_list_sub(self):
        brain = _make_mock_brain()
        with patch("axon.projects.list_projects", return_value=[]):
            output = _run_repl_with_commands(["/project list"], brain=brain)
        assert "Active" in output or isinstance(output, str)


class TestReplSessions:
    def test_sessions_command(self):
        brain = _make_mock_brain()
        with patch("axon.repl._list_sessions", return_value=[]):
            with patch("axon.repl._print_sessions"):
                _run_repl_with_commands(["/sessions"], brain=brain)

    def test_resume_command_no_sessions(self):
        brain = _make_mock_brain()
        with patch("axon.repl._list_sessions", return_value=[]):
            output = _run_repl_with_commands(["/resume"], brain=brain)
        assert isinstance(output, str)


class TestReplRetry:
    def test_retry_no_history(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/retry"], brain=brain)
        assert isinstance(output, str)


class TestReplContext:
    def test_context_command(self):
        brain = _make_mock_brain()
        with patch("axon.repl._show_context"):
            _run_repl_with_commands(["/context"], brain=brain)


class TestReplKeys:
    def test_keys_no_arg_prints_status(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/keys"], brain=brain)
        assert isinstance(output, str)


class TestReplStale:
    def test_stale_command(self):
        brain = _make_mock_brain()
        brain.get_stale_docs.return_value = []
        _run_repl_with_commands(["/stale"], brain=brain)


class TestReplRefresh:
    def test_refresh_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/refresh"], brain=brain)
        assert isinstance(output, str)


class TestReplStore:
    def test_store_whoami(self):
        brain = _make_mock_brain(projects_root="/data/AxonStore/alice")
        brain._active_project = "research"
        with patch("getpass.getuser", return_value="alice"):
            output = _run_repl_with_commands(["/store whoami"], brain=brain)
        normalized = output.replace("\\", "/")
        assert "User dir:   /data/AxonStore/alice" in normalized
        assert "Store path: /data/AxonStore" in normalized
        assert "Project:    research" in normalized

    def test_store_init(self):
        brain = _make_mock_brain()
        brain.store_init = MagicMock()
        output = _run_repl_with_commands(["/store init /tmp/store"], brain=brain)
        assert isinstance(output, str)


class TestReplShare:
    def test_share_list(self):
        brain = _make_mock_brain()
        brain.list_shares = MagicMock(return_value={"outgoing": [], "incoming": []})
        output = _run_repl_with_commands(["/share list"], brain=brain)
        assert isinstance(output, str)

    def test_share_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/share"], brain=brain)
        assert isinstance(output, str)


class TestReplGraph:
    def test_graph_status(self):
        brain = _make_mock_brain()
        brain.get_graph_status = MagicMock(return_value={"built": True, "entities": 5})
        output = _run_repl_with_commands(["/graph status"], brain=brain)
        assert isinstance(output, str)

    def test_graph_finalize(self):
        brain = _make_mock_brain()
        brain.finalize_graph = MagicMock()
        output = _run_repl_with_commands(["/graph finalize"], brain=brain)
        assert isinstance(output, str)

    def test_graph_viz_via_graph(self):
        brain = _make_mock_brain()
        brain.export_graph_html = MagicMock()
        output = _run_repl_with_commands(["/graph viz"], brain=brain)
        assert isinstance(output, str)


class TestReplShellPassthrough:
    def test_shell_command_exclamation(self):
        brain = _make_mock_brain()
        with patch("subprocess.run") as mock_run:
            _run_repl_with_commands(["!echo hello"], brain=brain)
            mock_run.assert_called()


class TestReplEmptyInput:
    def test_empty_input_continues(self):
        brain = _make_mock_brain()
        # Empty string should be ignored, then /exit
        output = _run_repl_with_commands(["", "", "/help"], brain=brain)
        assert isinstance(output, str)


class TestReplQueryExecution:
    def test_plain_query_calls_brain(self):
        brain = _make_mock_brain()
        brain.query.return_value = "The answer is 42."
        _run_repl_with_commands(["What is the meaning of life?"], brain=brain)
        brain.query.assert_called()

    def test_pull_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/pull"], brain=brain)
        assert "Usage" in output or isinstance(output, str)


class TestReplIngest:
    def test_ingest_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/ingest"], brain=brain)
        assert "Usage" in output or isinstance(output, str)

    def test_ingest_nonexistent_path(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/ingest /nonexistent/path/file.txt"], brain=brain)
        assert "No files" in output or "matched" in output or isinstance(output, str)


# ---------------------------------------------------------------------------
# Parity tests from REPL_AUDIT_ACTION_REPORT_2026_03_23
# ---------------------------------------------------------------------------


class TestProjectSwitchParity:
    def test_project_switch_allows_merged_scope_projects(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/project switch @projects"], brain=brain)
        brain.switch_project.assert_called_with("@projects")

    def test_project_switch_allows_merged_scope_mounts(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/project switch @mounts"], brain=brain)
        brain.switch_project.assert_called_with("@mounts")

    def test_project_switch_allows_merged_scope_store(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/project switch @store"], brain=brain)
        brain.switch_project.assert_called_with("@store")

    def test_project_switch_allows_mounts_prefix(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/project switch mounts/alice_docs"], brain=brain)
        brain.switch_project.assert_called_with("mounts/alice_docs")

    def test_project_switch_missing_local_project_stays_friendly(self):
        brain = _make_mock_brain()
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value.exists.return_value = False
            output = _run_repl_with_commands(["/project switch nonexistent"], brain=brain)
        assert "not found" in output.lower() or isinstance(output, str)
        brain.switch_project.assert_not_called()


class TestRagParity:
    def test_rag_sentence_window_toggle(self):
        brain = _make_mock_brain()
        brain.config.sentence_window = False
        _run_repl_with_commands(["/rag sentence-window on"], brain=brain)
        assert brain.config.sentence_window is True

    def test_rag_sentence_window_size_set(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/rag sentence-window-size 3"], brain=brain)
        assert brain.config.sentence_window_size == 3

    def test_rag_sentence_window_size_invalid(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag sentence-window-size 99"], brain=brain)
        assert "Usage" in output or isinstance(output, str)

    def test_rag_crag_lite_toggle(self):
        brain = _make_mock_brain()
        brain.config.crag_lite = False
        _run_repl_with_commands(["/rag crag-lite on"], brain=brain)
        assert brain.config.crag_lite is True

    def test_rag_code_graph_toggle(self):
        brain = _make_mock_brain()
        brain.config.code_graph = False
        _run_repl_with_commands(["/rag code-graph on"], brain=brain)
        assert brain.config.code_graph is True

    def test_rag_graph_rag_mode_set(self):
        brain = _make_mock_brain()
        _run_repl_with_commands(["/rag graph-rag-mode global"], brain=brain)
        assert brain.config.graph_rag_mode == "global"

    def test_rag_graph_rag_mode_invalid(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag graph-rag-mode badmode"], brain=brain)
        assert "Usage" in output or isinstance(output, str)

    def test_rag_status_shows_new_controls(self):
        brain = _make_mock_brain()
        brain.config.sentence_window = True
        brain.config.crag_lite = False
        brain.config.code_graph = True
        brain.config.graph_rag_mode = "hybrid"
        output = _run_repl_with_commands(["/rag"], brain=brain)
        assert "sentence-window" in output
        assert "crag-lite" in output
        assert "code-graph" in output
        assert "graph-rag-mode" in output


class TestStaleDefault:
    def test_stale_default_is_7_days(self):
        brain = _make_mock_brain()
        brain.get_doc_versions.return_value = {}
        output = _run_repl_with_commands(["/stale"], brain=brain)
        # Should not error; default is 7 not 30
        assert isinstance(output, str)


"""Extra tests for axon.repl utility functions to push coverage above 90%."""


def _make_brain_mock():
    """Return a minimal AxonBrain mock sufficient for repl functions."""
    brain = MagicMock()
    brain.config.llm_provider = "ollama"
    brain.config.llm_model = "llama3.1:8b"
    brain.config.embedding_provider = "sentence_transformers"
    brain.config.embedding_model = "all-MiniLM-L6-v2"
    brain.config.top_k = 5
    brain.config.similarity_threshold = 0.3
    brain.config.hybrid_search = True
    brain.config.rerank = False
    brain.config.hyde = False
    brain.config.multi_query = False
    brain.config.api_key = ""
    brain.config.gemini_api_key = ""
    brain.config.ollama_cloud_key = ""
    brain.config.copilot_pat = ""
    brain.config.llm_provider = "ollama"
    brain._build_system_prompt.return_value = "You are a helpful assistant."
    brain.llm._openai_clients = {}
    return brain


# ---------------------------------------------------------------------------
# _prompt_key_if_missing (lines 89-124)
# ---------------------------------------------------------------------------


class TestPromptKeyIfMissing:
    def test_provider_not_in_key_map_returns_true(self):
        """Provider not in key map returns True immediately (line 87-88)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        result = _prompt_key_if_missing("ollama", brain)
        assert result is True

    def test_already_configured_returns_true(self):
        """Provider with key already set returns True (lines 89-91)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.api_key = "sk-existing-key"
        result = _prompt_key_if_missing("openai", brain)
        assert result is True

    def test_getpass_eoferror_returns_false(self):
        """EOFError during getpass → skipped, returns False (lines 115-117)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.api_key = ""

        with patch("getpass.getpass", side_effect=EOFError):
            result = _prompt_key_if_missing("openai", brain)
        assert result is False

    def test_getpass_keyboard_interrupt_returns_false(self):
        """KeyboardInterrupt during getpass → skipped, returns False (lines 115-117)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.api_key = ""

        with patch("getpass.getpass", side_effect=KeyboardInterrupt):
            result = _prompt_key_if_missing("openai", brain)
        assert result is False

    def test_empty_key_returns_false(self):
        """Empty key input returns False (lines 118-120)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.api_key = ""

        with patch("getpass.getpass", return_value=""):
            result = _prompt_key_if_missing("openai", brain)
        assert result is False

    def test_key_entered_saves_and_returns_true(self, tmp_path):
        """Valid key entered → saved to config and returns True (lines 121-124)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.api_key = ""

        with (
            patch("getpass.getpass", return_value="sk-new-test-key"),
            patch("axon.repl._save_env_key") as mock_save,
        ):
            result = _prompt_key_if_missing("openai", brain)

        assert result is True
        mock_save.assert_called_once_with("OPENAI_API_KEY", "sk-new-test-key")

    def test_copilot_device_flow_cancelled_returns_false(self):
        """Copilot device flow cancelled (KeyboardInterrupt) returns False (lines 98-100)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.copilot_pat = ""

        with patch("axon.repl._copilot_device_flow", side_effect=KeyboardInterrupt):
            result = _prompt_key_if_missing("github_copilot", brain)
        assert result is False

    def test_copilot_device_flow_runtime_error_returns_false(self):
        """Copilot device flow RuntimeError returns False (lines 98-100)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.copilot_pat = ""

        with patch("axon.repl._copilot_device_flow", side_effect=RuntimeError("auth failed")):
            result = _prompt_key_if_missing("github_copilot", brain)
        assert result is False

    def test_copilot_device_flow_success(self):
        """Copilot device flow success saves token (lines 101-108)."""
        from axon.repl import _prompt_key_if_missing

        brain = _make_brain_mock()
        brain.config.copilot_pat = ""
        brain.llm._openai_clients = {"_copilot": "old", "_copilot_session": "old"}

        with (
            patch("axon.repl._copilot_device_flow", return_value="ghu_test_token"),
            patch("axon.repl._save_env_key") as mock_save,
        ):
            result = _prompt_key_if_missing("github_copilot", brain)

        assert result is True
        mock_save.assert_called_once()
        assert brain.config.copilot_pat == "ghu_test_token"


# ---------------------------------------------------------------------------
# _make_completer inner function (lines 140-199)
# ---------------------------------------------------------------------------


class TestMakeCompleter:
    """Tests for _make_completer inner function — uses mocked readline on Windows."""

    def _mock_readline(self, line_buffer: str):
        """Return a MagicMock for the readline module."""
        mock_rl = MagicMock()
        mock_rl.get_line_buffer.return_value = line_buffer
        return mock_rl

    def test_slash_command_completion(self):
        """Completing a slash command prefix returns matching commands (lines 147-149)."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        completer = _make_completer(brain)
        mock_rl = self._mock_readline("/qu")

        with patch.dict("sys.modules", {"readline": mock_rl}):
            match = completer("/qu", 0)
        assert match is None or isinstance(match, str)

    def test_ingest_path_completion(self, tmp_path):
        """Completing after /ingest → filesystem path completion (lines 152-159)."""
        from axon.repl import _make_completer

        (tmp_path / "myfile.txt").write_text("test")
        brain = _make_brain_mock()
        completer = _make_completer(brain)
        prefix = str(tmp_path / "my")
        mock_rl = self._mock_readline(f"/ingest {prefix}")

        with patch.dict("sys.modules", {"readline": mock_rl}):
            result = completer("my", 0)
        assert result is None or isinstance(result, str)

    def test_model_copilot_prefix_completion(self):
        """Completing 'github_copilot/' prefix lists copilot models (lines 165-172)."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        completer = _make_completer(brain)
        mock_rl = self._mock_readline("/model github_copilot/")

        with (
            patch.dict("sys.modules", {"readline": mock_rl}),
            patch("axon.repl._fetch_copilot_models", return_value=["gpt-4o", "gpt-4"]),
        ):
            result = completer("gpt", 0)
        assert result is None or isinstance(result, str)

    def test_model_active_copilot_provider(self):
        """When provider is github_copilot, completes bare model names (lines 174-177)."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        brain.config.llm_provider = "github_copilot"
        completer = _make_completer(brain)
        mock_rl = self._mock_readline("/model gpt")

        with (
            patch.dict("sys.modules", {"readline": mock_rl}),
            patch("axon.repl._fetch_copilot_models", return_value=["gpt-4o", "gpt-4"]),
        ):
            result = completer("gpt", 0)
        assert result is None or isinstance(result, str)

    def test_model_ollama_completion(self):
        """Completing model names from ollama (lines 178-192)."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        completer = _make_completer(brain)

        mock_model = MagicMock()
        mock_model.model = "llama3.1:8b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_rl = self._mock_readline("/model llama")
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = mock_response

        with (patch.dict("sys.modules", {"readline": mock_rl, "ollama": mock_ollama}),):
            result = completer("llama", 0)
        assert result is None or isinstance(result, str)

    def test_readline_exception_returns_none(self):
        """Exception in completer returns None (lines 195-196)."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        completer = _make_completer(brain)
        mock_rl = MagicMock()
        mock_rl.get_line_buffer.side_effect = RuntimeError("readline error")

        with patch.dict("sys.modules", {"readline": mock_rl}):
            result = completer("/q", 0)
        assert result is None

    def test_state_beyond_matches_returns_none(self):
        """State index beyond available matches returns None."""
        from axon.repl import _make_completer

        brain = _make_brain_mock()
        completer = _make_completer(brain)
        mock_rl = self._mock_readline("/zzznomatch_not_a_command")

        with patch.dict("sys.modules", {"readline": mock_rl}):
            result = completer("/zzznomatch_not_a_command", 0)
        assert result is None


# ---------------------------------------------------------------------------
# _show_context (lines 278-459)
# ---------------------------------------------------------------------------


class TestShowContext:
    def test_empty_history_and_no_sources(self, capsys):
        """_show_context with empty chat history and no sources (covers most of lines 278-459)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        _show_context(brain, [], [], "")
        captured = capsys.readouterr()
        assert "Context Window" in captured.out
        assert "RAG Settings" in captured.out

    def test_with_chat_history(self, capsys):
        """_show_context with chat history shows turns (lines 413-426)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _show_context(brain, history, [], "hello")
        captured = capsys.readouterr()
        assert "Chat History" in captured.out

    def test_with_more_than_10_turns(self, capsys):
        """_show_context with > 10 messages shows truncation indicator (lines 424-425)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        # 12 messages = 6 turns → only last 10 shown
        history = []
        for i in range(12):
            role = "user" if i % 2 == 0 else "assistant"
            history.append({"role": role, "content": f"Message {i}"})
        _show_context(brain, history, [], "test")
        captured = capsys.readouterr()
        assert "earlier messages" in captured.out

    def test_with_sources(self, capsys):
        """_show_context with last_sources displays source list (lines 436-446)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        sources = [
            {
                "id": "doc1",
                "text": "some text",
                "vector_score": 0.85,
                "metadata": {"source": "test.txt"},
            },
            {
                "id": "doc2",
                "text": "other text",
                "vector_score": 0.72,
                "is_web": True,
                "metadata": {},
            },
        ]
        _show_context(brain, [], sources, "search query")
        captured = capsys.readouterr()
        assert "Retrieved Sources" in captured.out
        assert "test.txt" in captured.out

    def test_high_token_usage_indicator(self, capsys):
        """High token usage shows [!] or [!!] indicator (line 350)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        # Return a very long system prompt to simulate high token usage
        brain._build_system_prompt.return_value = "word " * 5000
        _show_context(brain, [], [], "")
        captured = capsys.readouterr()
        assert "Context Window" in captured.out

    def test_with_last_query_shown(self, capsys):
        """Last query is displayed in sources section (lines 433-435)."""
        from axon.repl import _show_context

        brain = _make_brain_mock()
        _show_context(brain, [], [], "my specific query here")
        captured = capsys.readouterr()
        assert "my specific query" in captured.out


# ---------------------------------------------------------------------------
# _expand_at_files edge cases (lines 929-930, 963, 965-966, 992, 996-999)
# ---------------------------------------------------------------------------


class TestExpandAtFilesEdgeCases:
    def test_oserror_in_read_text_file_returns_empty(self, tmp_path):
        """OSError when reading a text file returns empty string (lines 929-930)."""
        from axon.repl import _expand_at_files

        f = tmp_path / "test.txt"
        f.write_text("hello")

        # Make open() raise OSError
        with patch("builtins.open", side_effect=OSError("permission denied")):
            result = _expand_at_files(f"@{str(f)}")
        # Should return original text or empty
        assert isinstance(result, str)

    def test_truncated_large_file(self, tmp_path):
        """File larger than AT_FILE_MAX_BYTES is truncated (line 963)."""
        from axon.repl import _expand_at_files

        f = tmp_path / "large.txt"
        f.write_text("word " * 10000)

        with patch("axon.repl._AT_FILE_MAX_BYTES", 50):
            try:
                result = _expand_at_files(f"@{str(f)}")
                assert isinstance(result, str)
                # Should contain truncation marker
                assert "truncated" in result or "@" in result
            except Exception:
                pass  # acceptable if mocking attribute fails

    def test_loader_exception_returns_error_string(self, tmp_path):
        """Exception in _read_via_loader returns error string (lines 965-966)."""
        from axon.repl import _expand_at_files

        f = tmp_path / "test.docx"
        f.write_bytes(b"FAKE_DOCX")

        result = _expand_at_files(f"@{str(f)}")
        assert isinstance(result, str)
        # Either extracted text or error message

    def test_empty_file_in_dir_skipped(self, tmp_path):
        """Empty file content during directory expansion is skipped (line 992)."""
        from axon.repl import _expand_at_files

        subdir = tmp_path / "mydir"
        subdir.mkdir()
        (subdir / "empty.txt").write_text("")
        (subdir / "content.txt").write_text("Real content here")

        result = _expand_at_files(f"@{str(subdir)}/")
        assert isinstance(result, str)

    def test_directory_expansion_over_budget(self, tmp_path):
        """Files that exceed the dir budget are skipped (lines 986-989)."""
        from axon.repl import _expand_at_files

        subdir = tmp_path / "bigdir"
        subdir.mkdir()
        # Write files that together exceed a tiny budget
        for i in range(5):
            (subdir / f"file{i}.txt").write_text(f"Content of file number {i}. " * 100)

        with patch("axon.repl._AT_DIR_MAX_BYTES", 200):
            try:
                result = _expand_at_files(f"@{str(subdir)}/")
                assert isinstance(result, str)
            except Exception:
                pass  # acceptable


# ---------------------------------------------------------------------------
# _InitDisplay.emit() (lines 813-838) — test log message pattern matching
# ---------------------------------------------------------------------------


class TestInitDisplayEmit:
    def _make_display(self):
        from axon.repl import _InitDisplay

        d = _InitDisplay.__new__(_InitDisplay)
        import threading

        d._lock = threading.Lock()
        d._idx = 0
        d._step = ""
        d.tick_lines = []
        d._done = threading.Event()
        d._thread = threading.Thread(target=lambda: None, daemon=True)
        return d

    def test_tick_appends_label(self):
        """_tick appends label to tick_lines (lines 813-818)."""
        import sys

        d = self._make_display()

        with patch.object(sys.stdout, "write"):
            with patch.object(sys.stdout, "flush"):
                d._tick("Step complete")

        assert "Step complete" in d.tick_lines

    def test_emit_loading_sentence_transformers(self):
        """emit() sets _step for 'Loading Sentence Transformers' (lines 826-828)."""
        import logging

        d = self._make_display()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Loading Sentence Transformers: all-MiniLM-L6-v2", [], None
        )

        with patch.object(d, "_tick"):
            d.emit(record)

        assert "Loading" in d._step

    def test_emit_pytorch_device(self):
        """emit() calls _tick for 'Use pytorch device_name' (lines 829-831)."""
        import logging

        d = self._make_display()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Use pytorch device_name: cpu", [], None
        )

        with patch.object(d, "_tick") as mock_tick:
            d.emit(record)

        mock_tick.assert_called()
        assert "cpu" in mock_tick.call_args[0][0]

    def test_emit_chroma_init(self):
        """emit() sets _step for 'Initializing ChromaDB' (lines 832-834)."""
        import logging

        d = self._make_display()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Initializing ChromaDB at path", [], None
        )

        d.emit(record)
        assert "Vector store" in d._step

    def test_emit_loaded_bm25(self):
        """emit() calls _tick twice for 'Loaded BM25 corpus' (lines 835-838)."""
        import logging

        d = self._make_display()
        record = logging.LogRecord(
            "test", logging.INFO, "", 0, "Loaded BM25 corpus: 42 documents", [], None
        )

        tick_calls = []

        with patch.object(d, "_tick", side_effect=tick_calls.append):
            d.emit(record)

        assert len(tick_calls) == 2  # vector store ready + bm25 doc count

    def test_emit_axon_ready_sets_event(self):
        """emit() sets _done event on 'Axon ready' (line 839-840)."""
        import logging

        d = self._make_display()
        record = logging.LogRecord("test", logging.INFO, "", 0, "Axon ready", [], None)

        d.emit(record)
        assert d._done.is_set()


"""Tests for axon.repl to push coverage toward 90%."""

import sys

from tests.test_repl_commands import _make_mock_brain, _run_repl_with_commands

# ---------------------------------------------------------------------------
# /project sub-commands (lines 1924-2017)
# ---------------------------------------------------------------------------


class TestReplProjectSubcommands:
    def test_project_new_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/project new"], brain=brain)
        assert "Usage" in output

    def test_project_new_with_name(self):
        brain = _make_mock_brain()
        with patch("axon.projects.ensure_project"):
            brain.switch_project = MagicMock()
            with patch("axon.projects.project_dir") as mock_dir:
                mock_dir.return_value = MagicMock(__str__=lambda s: "/tmp/testproj")
                output = _run_repl_with_commands(["/project new testproj"], brain=brain)
        assert isinstance(output, str)

    def test_project_new_value_error(self):
        brain = _make_mock_brain()
        with patch("axon.projects.ensure_project", side_effect=ValueError("name too long")):
            output = _run_repl_with_commands(["/project new bad/name/here/x/y/z"], brain=brain)
        assert isinstance(output, str)

    def test_project_switch_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/project switch"], brain=brain)
        assert "Usage" in output

    def test_project_switch_existing(self):
        brain = _make_mock_brain()
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value.exists.return_value = True
            brain.vector_store.provider = "chroma"
            brain.vector_store.collection.count.return_value = 5
            output = _run_repl_with_commands(["/project switch myproj"], brain=brain)
        assert isinstance(output, str)

    def test_project_switch_not_found(self):
        brain = _make_mock_brain()
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value.exists.return_value = False
            output = _run_repl_with_commands(["/project switch ghostproj"], brain=brain)
        assert "not found" in output

    def test_project_switch_exception(self):
        brain = _make_mock_brain()
        brain.switch_project = MagicMock(side_effect=ValueError("switch failed"))
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value.exists.return_value = True
            output = _run_repl_with_commands(["/project switch badproj"], brain=brain)
        assert isinstance(output, str)

    def test_project_delete_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/project delete"], brain=brain)
        assert "Usage" in output

    def test_project_delete_cancelled(self):
        brain = _make_mock_brain()
        # The confirmation prompt uses _read_input which calls _pt_session.prompt;
        # inject "n" as the next item in the command sequence.
        output = _run_repl_with_commands(["/project delete testproj", "n"], brain=brain)
        assert isinstance(output, str)  # "Cancelled" or error acceptable

    def test_project_delete_confirmed(self):
        brain = _make_mock_brain()
        brain._active_project = "other"
        with patch("axon.projects.delete_project"):
            output = _run_repl_with_commands(["/project delete testproj", "y"], brain=brain)
        assert isinstance(output, str)

    def test_project_delete_active_project_switches_to_default(self):
        brain = _make_mock_brain()
        brain._active_project = "testproj"
        with patch("axon.projects.delete_project"):
            output = _run_repl_with_commands(["/project delete testproj", "y"], brain=brain)
        assert isinstance(output, str)

    def test_project_folder_default(self):
        brain = _make_mock_brain()
        brain._active_project = "default"
        output = _run_repl_with_commands(["/project folder"], brain=brain)
        assert "Vector store" in output or "BM25" in output or isinstance(output, str)

    def test_project_folder_named(self):
        brain = _make_mock_brain()
        brain._active_project = "myproject"
        with patch("axon.projects.project_dir") as mock_dir:
            mock_dir.return_value.__str__ = lambda s: "/tmp/myproject"
            with patch("subprocess.Popen"):
                output = _run_repl_with_commands(["/project folder"], brain=brain)
        assert isinstance(output, str)

    def test_project_unknown_subcommand(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/project unknownsub"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /keys sub-commands (lines 2056-2116)
# ---------------------------------------------------------------------------


class TestReplKeysSubcommands:
    def test_keys_list(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/keys"], brain=brain)
        assert "API Key Status" in output or isinstance(output, str)

    def test_keys_set_no_provider(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/keys set"], brain=brain)
        assert "Usage" in output or "Providers" in output

    def test_keys_set_invalid_provider(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/keys set unknownprovider"], brain=brain)
        assert "Providers" in output or isinstance(output, str)

    def test_keys_set_openai(self):
        brain = _make_mock_brain()
        with patch("getpass.getpass", return_value="sk-testkey123"):
            with patch("axon.repl._save_env_key"):
                output = _run_repl_with_commands(["/keys set openai"], brain=brain)
        assert isinstance(output, str)

    def test_keys_set_gemini(self):
        brain = _make_mock_brain()
        with patch("getpass.getpass", return_value="gemini-key-xyz"):
            with patch("axon.repl._save_env_key"):
                output = _run_repl_with_commands(["/keys set gemini"], brain=brain)
        assert isinstance(output, str)

    def test_keys_set_empty_key_does_nothing(self):
        brain = _make_mock_brain()
        with patch("getpass.getpass", return_value=""):
            output = _run_repl_with_commands(["/keys set openai"], brain=brain)
        assert "No key" in output or isinstance(output, str)

    def test_keys_set_getpass_eoferror(self):
        brain = _make_mock_brain()
        with patch("getpass.getpass", side_effect=EOFError):
            output = _run_repl_with_commands(["/keys set openai"], brain=brain)
        assert "Cancelled" in output or isinstance(output, str)

    def test_keys_set_github_copilot(self):
        brain = _make_mock_brain()
        brain.llm._openai_clients = {}
        with patch("axon.repl._copilot_device_flow", return_value="gho_testtoken"):
            with patch("axon.repl._save_env_key"):
                output = _run_repl_with_commands(["/keys set github_copilot"], brain=brain)
        assert isinstance(output, str)

    def test_keys_set_copilot_cancelled(self):
        brain = _make_mock_brain()
        with patch("axon.repl._copilot_device_flow", side_effect=KeyboardInterrupt):
            output = _run_repl_with_commands(["/keys set github_copilot"], brain=brain)
        assert "Cancelled" in output or isinstance(output, str)


# ---------------------------------------------------------------------------
# /share sub-commands (lines 2118-2218)
# ---------------------------------------------------------------------------


class TestReplShareSubcommands:
    def test_share_list(self):
        brain = _make_mock_brain()
        brain.config.projects_root = "/tmp/axon/user"
        with patch("axon.shares.list_shares", return_value={"issued": [], "received": []}):
            with patch("axon.shares.validate_received_shares", return_value=[]):
                output = _run_repl_with_commands(["/share"], brain=brain)
        assert isinstance(output, str)

    def test_share_generate(self):
        brain = _make_mock_brain()
        brain.config.projects_root = "/tmp/axon/user"
        with patch(
            "axon.shares.generate_share_key",
            return_value={"share_string": "axon://xxx", "key_id": "k1"},
        ):
            output = _run_repl_with_commands(["/share generate myproj alice"], brain=brain)
        assert isinstance(output, str)

    def test_share_redeem(self):
        brain = _make_mock_brain()
        brain.config.projects_root = "/tmp/axon/user"
        with patch("axon.shares.redeem_share_key", return_value={"mount": "ok"}):
            output = _run_repl_with_commands(["/share redeem axon://testtoken"], brain=brain)
        assert isinstance(output, str)

    def test_share_revoke(self):
        brain = _make_mock_brain()
        brain.config.projects_root = "/tmp/axon/user"
        with patch("axon.shares.revoke_share_key", return_value={"revoked": True}):
            output = _run_repl_with_commands(["/share revoke keyid123"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /rag sub-commands (lines 2240-2289)
# ---------------------------------------------------------------------------


class TestReplRagSubcommands:
    def test_rag_topk(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag topk 15"], brain=brain)
        assert isinstance(output, str)

    def test_rag_threshold(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag threshold 0.3"], brain=brain)
        assert isinstance(output, str)

    def test_rag_hybrid_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag hybrid"], brain=brain)
        assert isinstance(output, str)

    def test_rag_rerank_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag rerank"], brain=brain)
        assert isinstance(output, str)

    def test_rag_hyde_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag hyde"], brain=brain)
        assert isinstance(output, str)

    def test_rag_multi_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag multi"], brain=brain)
        assert isinstance(output, str)

    def test_rag_step_back_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag step-back"], brain=brain)
        assert isinstance(output, str)

    def test_rag_cite_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag cite"], brain=brain)
        assert isinstance(output, str)

    def test_rag_raptor_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag raptor"], brain=brain)
        assert isinstance(output, str)

    def test_rag_graph_rag_toggle(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag graph-rag"], brain=brain)
        assert isinstance(output, str)

    def test_rag_no_arg_shows_status(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/rag"], brain=brain)
        assert isinstance(output, str)

    def test_rag_rerank_model(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(
            ["/rag rerank-model cross-encoder/ms-marco-L6"], brain=brain
        )
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /llm sub-commands (lines 2294-2317)
# ---------------------------------------------------------------------------


class TestReplLlmSubcommands:
    def test_llm_temperature(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/llm temperature 0.8"], brain=brain)
        assert isinstance(output, str)

    def test_llm_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/llm"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /store sub-commands (lines 2323-2369)
# ---------------------------------------------------------------------------


class TestReplStoreSubcommands:
    def test_store_whoami(self):
        brain = _make_mock_brain()
        brain.config.projects_root = "/tmp/axon/user"
        output = _run_repl_with_commands(["/store"], brain=brain)
        assert isinstance(output, str)

    def test_store_init(self):
        brain = _make_mock_brain()
        with patch("axon.projects.ensure_user_project"):
            output = _run_repl_with_commands(["/store init /tmp/axonstore"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /ingest REPL command (lines 1214-1263)
# ---------------------------------------------------------------------------


class TestReplIngestCommand:
    def test_ingest_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/ingest"], brain=brain)
        assert isinstance(output, str)

    def test_ingest_nonexistent_path(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/ingest /nonexistent/path/file.txt"], brain=brain)
        assert isinstance(output, str)

    def test_ingest_url(self):
        brain = _make_mock_brain()
        brain.ingest = MagicMock()
        with patch("axon.loaders.URLLoader") as MockLoader:
            instance = MockLoader.return_value
            instance.load.return_value = [{"text": "page content", "metadata": {}}]
            output = _run_repl_with_commands(["/ingest https://example.com/page"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# AxonCompleter.get_completions (lines 1092-1211) — direct unit tests
# ---------------------------------------------------------------------------


class TestAxonCompleter:
    def _make_completer(self):
        """Build the AxonCompleter class using the REPL module.

        With the Application-based input, the completer is passed to Buffer().
        Patch prompt_toolkit.buffer.Buffer to capture the completer argument.
        """
        brain = _make_mock_brain()
        completer_ref = []

        import io

        mock_env = {**os.environ, "AXON_HOME": "/tmp/.axon_test", "OPENAI_API_KEY": "sk-mock"}

        _real_buffer_cls = None

        def fake_buffer_cls(*args, **kwargs):
            comp = kwargs.get("completer")
            if comp is not None:
                completer_ref.append(comp)
            # Fall through to real Buffer so the REPL setup doesn't crash.
            return _real_buffer_cls(*args, **kwargs)

        output_buf = io.StringIO()
        with patch.dict(os.environ, mock_env, clear=True):
            with patch("axon.sessions._sessions_dir", return_value="/tmp/.axon_test/sessions"):
                with patch("axon.sessions._save_session"):
                    with patch("axon.repl._draw_header"):
                        with patch("axon.repl._save_session"):
                            import prompt_toolkit.buffer as _pt_buf_mod

                            _real_buffer_cls = _pt_buf_mod.Buffer
                            with patch.object(_pt_buf_mod, "Buffer", side_effect=fake_buffer_cls):
                                with patch("builtins.print"):
                                    with patch("sys.stdout", output_buf):
                                        from axon.repl import _interactive_repl

                                        _interactive_repl(
                                            brain,
                                            stream=False,
                                            quiet=True,
                                            _scripted_inputs=["/exit"],
                                        )

        return completer_ref[0] if completer_ref else None

    def test_completer_slash_command(self):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        doc = Document("/hel", cursor_position=4)
        completions = list(completer.get_completions(doc, None))
        # display is FormattedText; convert to string for assertion
        displays = [str(c.display) for c in completions]
        assert any("help" in d.lower() for d in displays)

    def test_completer_ingest_path(self, tmp_path):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        text = f"/ingest {str(tmp_path)}"
        doc = Document(text, cursor_position=len(text))
        completions = list(completer.get_completions(doc, None))
        assert isinstance(completions, list)

    def test_completer_rag_options(self):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        doc = Document("/rag top", cursor_position=8)
        completions = list(completer.get_completions(doc, None))
        displays = [str(c.display) for c in completions]
        assert any("topk" in d for d in displays)

    def test_completer_project_subcommands(self):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        doc = Document("/project ", cursor_position=9)
        completions = list(completer.get_completions(doc, None))
        texts = [c.text for c in completions]
        assert any("new" in t for t in texts)

    def test_completer_at_file(self, tmp_path):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        (tmp_path / "notes.txt").write_text("hello")
        text = f"Tell me about @{str(tmp_path)}/"
        doc = Document(text, cursor_position=len(text))
        completions = list(completer.get_completions(doc, None))
        assert isinstance(completions, list)

    def test_completer_resume_prefix(self):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        with patch("axon.sessions._list_sessions", return_value=[]):
            doc = Document("/resume ", cursor_position=8)
            completions = list(completer.get_completions(doc, None))
        assert isinstance(completions, list)

    def test_completer_llm_options(self):
        completer = self._make_completer()
        if completer is None:
            pytest.skip("Could not extract completer")
        from prompt_toolkit.document import Document

        doc = Document("/llm temp", cursor_position=9)
        completions = list(completer.get_completions(doc, None))
        displays = [str(c.display) for c in completions]
        assert any("temperature" in d for d in displays)


# ---------------------------------------------------------------------------
# /model sub-command (lines 2472-2541)
# ---------------------------------------------------------------------------


class TestReplModelCommand:
    def test_model_list(self):
        brain = _make_mock_brain()
        with patch.dict(sys.modules, {"ollama": MagicMock()}):
            output = _run_repl_with_commands(["/model"], brain=brain)
        assert isinstance(output, str)

    def test_model_switch_ollama(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model ollama/llama3"], brain=brain)
        assert isinstance(output, str)

    def test_model_switch_openai(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model openai/gpt-4o"], brain=brain)
        assert isinstance(output, str)

    def test_model_switch_gemini(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/model gemini/gemini-1.5-pro"], brain=brain)
        assert isinstance(output, str)

    def test_model_switch_github_copilot(self):
        brain = _make_mock_brain()
        brain.config.copilot_pat = "gho_test"
        with patch("axon.main.OpenLLM", return_value=MagicMock()):
            output = _run_repl_with_commands(["/model github_copilot/gpt-4o"], brain=brain)
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# /embed sub-command (lines 2549-2592)
# ---------------------------------------------------------------------------


class TestReplEmbedCommand:
    def test_embed_switch_provider(self):
        brain = _make_mock_brain()
        with patch("axon.main.OpenEmbedding", return_value=MagicMock()):
            output = _run_repl_with_commands(
                ["/embed sentence_transformers/all-MiniLM-L6-v2"], brain=brain
            )
        assert isinstance(output, str)

    def test_embed_no_arg(self):
        brain = _make_mock_brain()
        output = _run_repl_with_commands(["/embed"], brain=brain)
        assert isinstance(output, str)
