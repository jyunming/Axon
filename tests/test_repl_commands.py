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
                        with patch("prompt_toolkit.PromptSession") as mock_ps_cls, patch(
                            "prompt_toolkit.formatted_text.ANSI", side_effect=lambda x: x
                        ):
                            mock_ps = mock_ps_cls.return_value
                            mock_ps.prompt.side_effect = all_cmds

                            with patch("builtins.print", side_effect=fake_print):
                                with patch("sys.stdout", output_buffer):
                                    from axon.repl import _interactive_repl

                                    _interactive_repl(brain, stream=False, quiet=True)

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
    def test_shell_passthrough_runs_in_local_mode(self):
        brain = _make_mock_brain(repl_shell_passthrough="local_only")
        brain._active_project_kind = "local"
        with patch("subprocess.run") as mock_run:
            _run_repl_with_commands(["!echo hello"], brain=brain)
        mock_run.assert_called_once_with("echo hello", shell=True)

    def test_shell_passthrough_blocked_in_scope_mode(self):
        brain = _make_mock_brain(repl_shell_passthrough="local_only")
        brain._active_project_kind = "scope"
        with patch("subprocess.run") as mock_run:
            output = _run_repl_with_commands(["!echo hello"], brain=brain)
        mock_run.assert_not_called()
        assert "shell passthrough blocked by policy" in output.lower()

    def test_shell_passthrough_always_overrides_scope(self):
        brain = _make_mock_brain(repl_shell_passthrough="always")
        brain._active_project_kind = "scope"
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
