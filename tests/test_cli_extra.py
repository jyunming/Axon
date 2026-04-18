"""
tests/test_cli_extra.py

Comprehensive pytest tests for src/axon/cli.py.

Covers:
  - _print_project_tree() helper
  - _write_python_discovery() helper
  - main() via patched sys.argv for all major CLI paths

All imports inside main() are lazy, so we patch at the source module level:
  - AxonBrain  → axon.main.AxonBrain
  - AxonConfig → axon.config.AxonConfig
  - REPL helpers → axon.repl._interactive_repl / axon.repl._InitDisplay
  - Project functions → axon.projects.*
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: run main() with given args, capture output
# ---------------------------------------------------------------------------


def run_cli(*args):
    """Run main() with given args, return exit code (stdout captured by capsys)."""
    with patch("sys.argv", ["axon"] + list(args)):
        try:
            from axon.cli import main

            main()
        except SystemExit as e:
            return e.code
        return 0


# ---------------------------------------------------------------------------
# Shared mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_brain_instance():
    """Return a pre-configured MagicMock for AxonBrain."""
    m = MagicMock()
    m.query.return_value = "test answer"
    m.query_stream.return_value = iter(["chunk1", "chunk2"])
    m.list_documents.return_value = []
    m.config = MagicMock()
    m._active_project = "default"
    return m


@pytest.fixture()
def mock_config():
    """Return a MagicMock config object so attribute assignments work."""
    cfg = MagicMock()
    cfg.llm_provider = "ollama"
    cfg.llm_model = "gemma:2b"
    cfg.vllm_base_url = "http://localhost:8000"
    return cfg


@pytest.fixture(autouse=True)
def _patch_all(mock_brain_instance, mock_config):
    """Patch all heavy components used by main() in every test."""
    with (
        patch("axon.main.AxonBrain", return_value=mock_brain_instance),
        patch("axon.config.AxonConfig.load", return_value=mock_config),
        patch("axon.repl._interactive_repl") as _repl,
        patch("axon.repl._InitDisplay", return_value=MagicMock()),
        patch("axon.repl._infer_provider", side_effect=lambda m: "ollama"),
        patch("sys.stdin.isatty", return_value=False),
        # Prevent hard process exit at end of REPL path
        patch("os._exit"),
        # Prevent auto-pull HTTP requests to local Ollama
        patch("ollama.list", return_value=MagicMock(models=[])),
    ):
        yield _repl


# ---------------------------------------------------------------------------
# Convenience re-exports of fixtures for tests that need them explicitly
# ---------------------------------------------------------------------------


@pytest.fixture()
def brain(mock_brain_instance):
    return mock_brain_instance


@pytest.fixture()
def cfg(mock_config):
    return mock_config


# ---------------------------------------------------------------------------
# 1.  _print_project_tree()
# ---------------------------------------------------------------------------


class TestPrintProjectTree:
    """Unit tests for _print_project_tree() — no main() invocation needed."""

    def _capture(self, proj_list, active, indent=0) -> str:
        from axon.cli import _print_project_tree

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_project_tree(proj_list, active, indent)
        return buf.getvalue()

    def _proj(self, name, created_at="2024-01-15T00:00:00", **kwargs):
        return {"name": name, "created_at": created_at, **kwargs}

    # --- basic rendering ---

    def test_active_marker_shown(self):
        out = self._capture([self._proj("myproj")], active="myproj")
        assert "●" in out

    def test_inactive_marker_space(self):
        out = self._capture([self._proj("myproj")], active="other")
        assert "●" not in out

    def test_timestamp_trimmed_to_date(self):
        out = self._capture([self._proj("myproj", created_at="2024-06-30T12:34:56")], active="")
        assert "2024-06-30" in out

    def test_missing_timestamp_renders_empty(self):
        p = self._proj("myproj", created_at=None)
        out = self._capture([p], active="")
        assert "myproj" in out  # should not raise

    def test_description_rendered(self):
        p = self._proj("myproj", description="My cool project")
        out = self._capture([p], active="")
        assert "My cool project" in out

    def test_no_description_no_extra_text(self):
        p = self._proj("myproj")
        out = self._capture([p], active="")
        assert "None" not in out

    def test_merged_tag_shown_when_children_present(self):
        child = self._proj("parent/child")
        parent = self._proj("parent", children=[child])
        out = self._capture([parent], active="")
        assert "[merged]" in out

    def test_no_merged_tag_without_children(self):
        p = self._proj("myproj")
        out = self._capture([p], active="")
        assert "[merged]" not in out

    def test_maintenance_state_shown(self):
        p = self._proj("myproj", maintenance_state="archived")
        out = self._capture([p], active="")
        assert "[archived]" in out

    def test_normal_state_not_shown(self):
        p = self._proj("myproj", maintenance_state="normal")
        out = self._capture([p], active="")
        assert "[normal]" not in out

    def test_missing_maintenance_state_no_bracket(self):
        p = self._proj("myproj")
        out = self._capture([p], active="")
        assert "[normal]" not in out

    def test_short_name_uses_last_segment(self):
        p = self._proj("research/papers/2024")
        out = self._capture([p], active="")
        assert "2024" in out

    def test_full_path_not_printed_as_short_name(self):
        p = self._proj("research/papers/2024")
        out = self._capture([p], active="")
        assert "research/papers/2024" not in out

    def test_children_recursed(self):
        child = self._proj("parent/child")
        parent = self._proj("parent", children=[child])
        out = self._capture([parent], active="")
        assert "child" in out

    def test_multiple_projects_listed(self):
        projects = [self._proj("alpha"), self._proj("beta")]
        out = self._capture(projects, active="alpha")
        assert "alpha" in out
        assert "beta" in out

    def test_empty_list_no_output(self):
        out = self._capture([], active="anything")
        assert out == ""

    def test_indent_applied_to_children(self):
        child = self._proj("parent/child")
        parent = self._proj("parent", children=[child])
        out = self._capture([parent], active="")
        lines = out.splitlines()
        parent_line = next(ln for ln in lines if "parent" in ln and "child" not in ln)
        child_line = next(ln for ln in lines if "child" in ln)
        parent_indent = len(parent_line) - len(parent_line.lstrip(" "))
        child_indent = len(child_line) - len(child_line.lstrip(" "))
        assert child_indent > parent_indent

    def test_active_matches_full_name(self):
        p = self._proj("research/papers")
        out_active = self._capture([p], active="research/papers")
        out_inactive = self._capture([p], active="research")
        assert "●" in out_active
        assert "●" not in out_inactive

    def test_description_only_when_nonempty(self):
        p_with = self._proj("a", description="has desc")
        p_without = self._proj("b")
        out_with = self._capture([p_with], active="")
        out_without = self._capture([p_without], active="")
        assert "has desc" in out_with
        # out_without should have no trailing description column text
        assert "  " in out_without  # just the padding — no crash

    def test_multiple_maintenance_states(self):
        p_maint = self._proj("mproj", maintenance_state="deprecated")
        out = self._capture([p_maint], active="")
        assert "[deprecated]" in out


# ---------------------------------------------------------------------------
# 2.  _write_python_discovery()
# ---------------------------------------------------------------------------


class TestWritePythonDiscovery:
    def test_writes_to_correct_location(self, tmp_path):
        from axon.cli import _write_python_discovery

        with patch("pathlib.Path.home", return_value=tmp_path):
            _write_python_discovery()
        expected = tmp_path / ".axon" / ".python_path"
        assert expected.exists()
        assert expected.read_text(encoding="utf-8") == sys.executable

    def test_silently_ignores_oserror(self):
        from axon.cli import _write_python_discovery

        with patch("pathlib.Path.home", side_effect=OSError("no home")):
            _write_python_discovery()  # must not raise

    def test_silently_ignores_permission_error(self, tmp_path):
        from axon.cli import _write_python_discovery

        bad_dir = tmp_path / "noperm"
        bad_dir.mkdir()

        def _bad_write(*a, **kw):
            raise PermissionError("no write")

        with (
            patch("pathlib.Path.home", return_value=tmp_path),
            patch("pathlib.Path.write_text", side_effect=_bad_write),
        ):
            _write_python_discovery()  # must not raise

    def test_creates_parent_directory(self, tmp_path):
        from axon.cli import _write_python_discovery

        with patch("pathlib.Path.home", return_value=tmp_path):
            _write_python_discovery()
        assert (tmp_path / ".axon").is_dir()


# ---------------------------------------------------------------------------
# 3.  main() — no args → enters REPL
# ---------------------------------------------------------------------------


class TestMainNoArgs:
    def test_no_args_calls_interactive_repl(self, _patch_all):
        _repl = _patch_all
        run_cli()
        _repl.assert_called_once()

    def test_no_args_repl_receives_brain(self, _patch_all, brain):
        _repl = _patch_all
        run_cli()
        call_args = _repl.call_args
        # brain is the first positional arg
        assert call_args[0][0] is brain

    def test_no_args_does_not_call_query(self, brain):
        run_cli()
        brain.query.assert_not_called()

    def test_no_args_constructs_brain_without_legacy_kwargs(self, mock_config, brain):
        with (
            patch("axon.main.AxonBrain", return_value=brain) as brain_ctor,
            patch("os._exit") as hard_exit,
        ):
            run_cli()
        brain_ctor.assert_called_once_with(mock_config)
        hard_exit.assert_not_called()


# ---------------------------------------------------------------------------
# 4.  main() --project-list
# ---------------------------------------------------------------------------


class TestMainProjectList:
    def test_project_list_empty_prints_message(self, capsys):
        with patch("axon.projects.list_projects", return_value=[]):
            run_cli("--project-list")
        assert "No projects yet" in capsys.readouterr().out

    def test_project_list_with_projects_prints_active(self, brain, capsys):
        brain._active_project = "myproject"
        projects = [{"name": "myproject", "created_at": "2024-01-01T00:00:00", "children": []}]
        with patch("axon.projects.list_projects", return_value=projects):
            run_cli("--project-list")
        out = capsys.readouterr().out
        assert "Active" in out

    def test_project_list_returns_early_without_query(self, brain):
        with patch("axon.projects.list_projects", return_value=[]):
            run_cli("--project-list")
        brain.query.assert_not_called()

    def test_project_list_shows_project_name(self, brain, capsys):
        brain._active_project = "work"
        projects = [{"name": "work", "created_at": "2024-06-01T00:00:00", "children": []}]
        with patch("axon.projects.list_projects", return_value=projects):
            run_cli("--project-list")
        assert "work" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5.  main() --project-delete
# ---------------------------------------------------------------------------


class TestMainProjectDelete:
    def test_delete_calls_delete_project(self, brain):
        brain._active_project = "other"
        with patch("axon.projects.delete_project") as _del:
            run_cli("--project-delete", "myproject")
        _del.assert_called_once_with("myproject")

    def test_delete_active_project_switches_to_default_first(self, brain):
        brain._active_project = "myproject"
        with patch("axon.projects.delete_project"):
            run_cli("--project-delete", "myproject")
        brain.switch_project.assert_called_with("default")

    def test_delete_lowercases_name(self, brain):
        brain._active_project = "other"
        with patch("axon.projects.delete_project") as _del:
            run_cli("--project-delete", "MyProject")
        _del.assert_called_once_with("myproject")

    def test_delete_has_children_error_exits_1(self, brain):
        brain._active_project = "other"
        from axon.projects import ProjectHasChildrenError

        with patch(
            "axon.projects.delete_project", side_effect=ProjectHasChildrenError("has children")
        ):
            code = run_cli("--project-delete", "myproject")
        assert code == 1

    def test_delete_value_error_exits_1(self, brain):
        brain._active_project = "other"
        with patch("axon.projects.delete_project", side_effect=ValueError("not found")):
            code = run_cli("--project-delete", "myproject")
        assert code == 1

    def test_delete_prints_confirmation(self, brain, capsys):
        brain._active_project = "other"
        with patch("axon.projects.delete_project"):
            run_cli("--project-delete", "myproject")
        out = capsys.readouterr().out
        assert "Deleted project" in out
        assert "myproject" in out

    def test_delete_nonactive_does_not_switch(self, brain):
        brain._active_project = "other"
        with patch("axon.projects.delete_project"):
            run_cli("--project-delete", "myproject")
        # switch_project should NOT have been called for "default"
        for c in brain.switch_project.call_args_list:
            assert c != call("default")


# ---------------------------------------------------------------------------
# 6.  main() --list
# ---------------------------------------------------------------------------


class TestMainList:
    def test_list_empty_knowledge_base(self, brain, capsys):
        brain.list_documents.return_value = []
        run_cli("--list")
        assert "empty" in capsys.readouterr().out.lower()

    def test_list_with_documents(self, brain, capsys):
        brain.list_documents.return_value = [
            {"source": "doc1.pdf", "chunks": 5},
            {"source": "doc2.md", "chunks": 3},
        ]
        run_cli("--list")
        out = capsys.readouterr().out
        assert "doc1.pdf" in out
        assert "doc2.md" in out
        assert "2 file(s)" in out
        assert "8 chunk(s)" in out

    def test_list_returns_early(self, brain):
        brain.list_documents.return_value = []
        run_cli("--list")
        brain.query.assert_not_called()

    def test_list_shows_chunk_counts(self, brain, capsys):
        brain.list_documents.return_value = [{"source": "file.txt", "chunks": 12}]
        run_cli("--list")
        assert "12" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 7.  main() --ingest
# ---------------------------------------------------------------------------


class TestMainIngest:
    def test_ingest_directory_calls_load_directory(self, brain, tmp_path):
        with patch("os.path.isdir", return_value=True), patch("asyncio.run") as _arun:
            run_cli("--ingest", str(tmp_path))
        _arun.assert_called_once()

    def test_ingest_file_calls_brain_ingest(self, brain, tmp_path):
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello world")
        fake_doc = {"text": "hello world", "metadata": {"type": "text"}}
        loader_mock = MagicMock()
        loader_mock.loaders = {".txt": MagicMock(load=MagicMock(return_value=[fake_doc]))}
        with (
            patch("os.path.isdir", return_value=False),
            patch("axon.loaders.DirectoryLoader", return_value=loader_mock),
        ):
            run_cli("--ingest", str(test_file))
        brain.ingest.assert_called_once()

    def test_ingest_adds_filepath_breadcrumb(self, brain, tmp_path):
        test_file = tmp_path / "doc.txt"
        test_file.write_text("hello")
        fake_doc = {"text": "hello", "metadata": {"type": "text"}}
        loader_mock = MagicMock()
        loader_mock.loaders = {".txt": MagicMock(load=MagicMock(return_value=[fake_doc]))}
        with (
            patch("os.path.isdir", return_value=False),
            patch("axon.loaders.DirectoryLoader", return_value=loader_mock),
        ):
            run_cli("--ingest", str(test_file))
        docs_passed = brain.ingest.call_args[0][0]
        assert any("[File Path:" in d["text"] for d in docs_passed)

    def test_ingest_skips_csv_breadcrumb(self, brain, tmp_path):
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b\n1,2")
        fake_doc = {"text": "a,b\n1,2", "metadata": {"type": "csv"}}
        loader_mock = MagicMock()
        loader_mock.loaders = {".csv": MagicMock(load=MagicMock(return_value=[fake_doc]))}
        with (
            patch("os.path.isdir", return_value=False),
            patch("axon.loaders.DirectoryLoader", return_value=loader_mock),
        ):
            run_cli("--ingest", str(test_file))
        docs_passed = brain.ingest.call_args[0][0]
        for doc in docs_passed:
            assert "[File Path:" not in doc["text"]

    def test_ingest_skips_image_breadcrumb(self, brain, tmp_path):
        test_file = tmp_path / "photo.jpg"
        test_file.write_bytes(b"\xff\xd8\xff")
        fake_doc = {"text": "", "metadata": {"type": "image"}}
        loader_mock = MagicMock()
        loader_mock.loaders = {".jpg": MagicMock(load=MagicMock(return_value=[fake_doc]))}
        with (
            patch("os.path.isdir", return_value=False),
            patch("axon.loaders.DirectoryLoader", return_value=loader_mock),
        ):
            run_cli("--ingest", str(test_file))
        docs_passed = brain.ingest.call_args[0][0]
        for doc in docs_passed:
            assert "[File Path:" not in doc["text"]


# ---------------------------------------------------------------------------
# 8.  main() --ingest + --project-new
# ---------------------------------------------------------------------------


class TestMainIngestWithProjectNew:
    def test_project_new_then_ingest(self, brain, tmp_path):
        with (
            patch("axon.projects.ensure_project") as _ensure,
            patch("axon.projects.project_dir", return_value=str(tmp_path)),
            patch("os.path.isdir", return_value=True),
            patch("asyncio.run"),
        ):
            run_cli("--project-new", "myproject", "--ingest", str(tmp_path))
        _ensure.assert_called_once_with("myproject")
        brain.switch_project.assert_called_with("myproject")

    def test_project_new_prints_project_path(self, brain, tmp_path, capsys):
        with (
            patch("axon.projects.ensure_project"),
            patch("axon.projects.project_dir", return_value=str(tmp_path)),
            patch("os.path.isdir", return_value=True),
            patch("asyncio.run"),
        ):
            run_cli("--project-new", "myproject", "--ingest", str(tmp_path))
        out = capsys.readouterr().out
        assert "myproject" in out

    def test_project_new_lowercases_name(self, brain, tmp_path):
        with (
            patch("axon.projects.ensure_project") as _ensure,
            patch("axon.projects.project_dir", return_value=str(tmp_path)),
        ):
            run_cli("--project-new", "MyNewProject", "test query")
        _ensure.assert_called_once_with("mynewproject")


# ---------------------------------------------------------------------------
# 9.  main() with positional query
# ---------------------------------------------------------------------------


class TestMainQuery:
    def test_query_calls_brain_query(self, brain):
        brain.query.return_value = "the answer"
        run_cli("what is axon?")
        brain.query.assert_called_once_with("what is axon?")

    def test_query_prints_response(self, brain, capsys):
        brain.query.return_value = "the answer"
        run_cli("what is axon?")
        assert "the answer" in capsys.readouterr().out

    def test_query_prints_response_label(self, brain, capsys):
        brain.query.return_value = "some text"
        run_cli("test question")
        assert "Response" in capsys.readouterr().out

    def test_query_returns_early_no_repl(self, _patch_all, brain):
        _repl = _patch_all
        brain.query.return_value = "answer"
        run_cli("my question")
        _repl.assert_not_called()


# ---------------------------------------------------------------------------
# 10.  main() --stream + query
# ---------------------------------------------------------------------------


class TestMainStream:
    def test_stream_calls_query_stream(self, brain):
        brain.query_stream.return_value = iter(["word1", " word2"])
        run_cli("--stream", "my question")
        brain.query_stream.assert_called_once_with("my question")

    def test_stream_prints_chunks(self, brain, capsys):
        brain.query_stream.return_value = iter(["hello", " world"])
        run_cli("--stream", "test")
        out = capsys.readouterr().out
        assert "hello" in out
        assert "world" in out

    def test_stream_skips_dict_chunks(self, brain, capsys):
        brain.query_stream.return_value = iter([{"meta": True}, "real text"])
        run_cli("--stream", "test")
        out = capsys.readouterr().out
        assert "real text" in out
        assert "meta" not in out

    def test_stream_without_query_does_not_stream(self, brain):
        run_cli("--stream")
        brain.query_stream.assert_not_called()


# ---------------------------------------------------------------------------
# 11.  main() --provider
# ---------------------------------------------------------------------------


class TestMainProvider:
    def test_provider_sets_llm_provider(self, cfg):
        run_cli("--provider", "gemini", "test query")
        assert cfg.llm_provider == "gemini"

    def test_provider_openai(self, cfg):
        run_cli("--provider", "openai", "test query")
        assert cfg.llm_provider == "openai"

    def test_provider_ollama(self, cfg):
        run_cli("--provider", "ollama", "test query")
        assert cfg.llm_provider == "ollama"

    def test_provider_vllm(self, cfg):
        run_cli("--provider", "vllm", "test query")
        assert cfg.llm_provider == "vllm"

    def test_provider_github_copilot(self, cfg):
        run_cli("--provider", "github_copilot", "test query")
        assert cfg.llm_provider == "github_copilot"


# ---------------------------------------------------------------------------
# 12.  main() --model (infer provider)
# ---------------------------------------------------------------------------


class TestMainModelInfer:
    def test_model_sets_llm_model(self, cfg):
        with patch("axon.repl._infer_provider", return_value="ollama"):
            run_cli("--model", "gemma:2b", "test query")
        assert cfg.llm_model == "gemma:2b"

    def test_model_infer_called(self, cfg):
        with patch("axon.repl._infer_provider", return_value="gemini") as _inf:
            run_cli("--model", "gemini-1.5-flash", "test query")
        _inf.assert_called_once_with("gemini-1.5-flash")

    def test_model_infer_provider_applied(self, cfg):
        with patch("axon.repl._infer_provider", return_value="gemini"):
            run_cli("--model", "gemini-1.5-flash", "test query")
        assert cfg.llm_provider == "gemini"
        assert cfg.llm_model == "gemini-1.5-flash"


# ---------------------------------------------------------------------------
# 13.  main() --model provider/model (slash-split)
# ---------------------------------------------------------------------------


class TestMainModelSlashSplit:
    def test_known_provider_prefix_splits_openai(self, cfg):
        run_cli("--model", "openai/gpt-4o", "test query")
        assert cfg.llm_provider == "openai"
        assert cfg.llm_model == "gpt-4o"

    def test_known_provider_prefix_splits_gemini(self, cfg):
        run_cli("--model", "gemini/gemini-2.0-flash", "test query")
        assert cfg.llm_provider == "gemini"
        assert cfg.llm_model == "gemini-2.0-flash"

    def test_known_provider_prefix_splits_ollama(self, cfg):
        run_cli("--model", "ollama/llama3.1", "test query")
        assert cfg.llm_provider == "ollama"
        assert cfg.llm_model == "llama3.1"

    def test_unknown_prefix_falls_back_to_infer(self, cfg):
        with patch("axon.repl._infer_provider", return_value="ollama") as _inf:
            run_cli("--model", "custom/mymodel", "test query")
        _inf.assert_called_once_with("custom/mymodel")
        assert cfg.llm_model == "custom/mymodel"


# ---------------------------------------------------------------------------
# 14.  main() --temperature
# ---------------------------------------------------------------------------


class TestMainTemperature:
    def test_temperature_sets_config(self, cfg):
        run_cli("--temperature", "0.5", "test query")
        assert cfg.llm_temperature == pytest.approx(0.5)

    def test_temperature_zero(self, cfg):
        run_cli("--temperature", "0.0", "test query")
        assert cfg.llm_temperature == pytest.approx(0.0)

    def test_temperature_high(self, cfg):
        run_cli("--temperature", "1.8", "test query")
        assert cfg.llm_temperature == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# 15.  main() --top-k
# ---------------------------------------------------------------------------


class TestMainTopK:
    def test_top_k_sets_config(self, cfg):
        run_cli("--top-k", "5", "test query")
        assert cfg.top_k == 5

    def test_top_k_large_value(self, cfg):
        run_cli("--top-k", "50", "test query")
        assert cfg.top_k == 50


# ---------------------------------------------------------------------------
# 16.  main() --list-models
# ---------------------------------------------------------------------------


class TestMainListModels:
    def test_list_models_prints_providers(self, capsys):
        with patch("ollama.list", return_value=MagicMock(models=[])):
            run_cli("--list-models")
        out = capsys.readouterr().out
        assert "ollama" in out
        assert "gemini" in out
        assert "openai" in out

    def test_list_models_returns_early_no_query(self, brain):
        with patch("ollama.list", return_value=MagicMock(models=[])):
            run_cli("--list-models")
        brain.query.assert_not_called()

    def test_list_models_ollama_not_reachable(self, capsys):
        with patch("ollama.list", side_effect=Exception("connection refused")):
            run_cli("--list-models")
        out = capsys.readouterr().out
        assert "not reachable" in out.lower() or "Ollama" in out

    def test_list_models_shows_local_model_names(self, capsys):
        model_mock = MagicMock()
        model_mock.model = "gemma:2b"
        model_mock.size = 2_000_000_000
        response = MagicMock()
        response.models = [model_mock]
        with patch("ollama.list", return_value=response):
            run_cli("--list-models")
        assert "gemma:2b" in capsys.readouterr().out

    def test_list_models_shows_size_in_gb(self, capsys):
        model_mock = MagicMock()
        model_mock.model = "llama3.1"
        model_mock.size = 4_500_000_000
        response = MagicMock()
        response.models = [model_mock]
        with patch("ollama.list", return_value=response):
            run_cli("--list-models")
        out = capsys.readouterr().out
        assert "GB" in out or "4.5" in out


# ---------------------------------------------------------------------------
# 17.  main() --pull MODEL
# ---------------------------------------------------------------------------


class TestMainPull:
    def test_pull_calls_ollama_pull(self):
        chunk = MagicMock(status="downloading", total=100, completed=50)
        with patch("ollama.pull", return_value=iter([chunk])) as _pull:
            run_cli("--pull", "gemma:2b")
        _pull.assert_called_once_with("gemma:2b", stream=True)

    def test_pull_prints_ready_message(self, capsys):
        chunk = MagicMock(status="done", total=0, completed=0)
        with patch("ollama.pull", return_value=iter([chunk])):
            run_cli("--pull", "gemma:2b")
        assert "ready" in capsys.readouterr().out.lower()

    def test_pull_returns_early_no_query(self, brain):
        chunk = MagicMock(status="done", total=0, completed=0)
        with patch("ollama.pull", return_value=iter([chunk])):
            run_cli("--pull", "gemma:2b")
        brain.query.assert_not_called()

    def test_pull_error_prints_error_message(self, capsys):
        with patch("ollama.pull", side_effect=Exception("failed")):
            run_cli("--pull", "gemma:2b")
        assert "Error" in capsys.readouterr().out

    def test_pull_dict_chunk_with_progress(self, capsys):
        chunk = {"status": "downloading", "total": 100, "completed": 75}
        with patch("ollama.pull", return_value=iter([chunk])):
            run_cli("--pull", "gemma:2b")
        out = capsys.readouterr().out
        assert "75%" in out or "gemma:2b" in out

    def test_pull_dict_chunk_status_only(self, capsys):
        chunk = {"status": "success", "total": 0, "completed": 0}
        with patch("ollama.pull", return_value=iter([chunk])):
            run_cli("--pull", "mymodel")
        out = capsys.readouterr().out
        assert "success" in out or "ready" in out.lower()

    def test_pull_obj_chunk_with_progress(self, capsys):
        chunk = MagicMock(status="pulling", total=200, completed=100)
        with patch("ollama.pull", return_value=iter([chunk])):
            run_cli("--pull", "phi3")
        out = capsys.readouterr().out
        assert "50%" in out or "pulling" in out


# ---------------------------------------------------------------------------
# 18.  main() --project NAME
# ---------------------------------------------------------------------------


class TestMainProject:
    def test_project_switch_calls_switch_project(self, brain):
        run_cli("--project", "myproject", "test query")
        brain.switch_project.assert_called_with("myproject")

    def test_project_lowercased(self, brain):
        run_cli("--project", "MyProject", "test query")
        brain.switch_project.assert_called_with("myproject")

    def test_project_switch_value_error_exits_1(self, brain):
        brain.switch_project.side_effect = ValueError("not found")
        code = run_cli("--project", "badproject", "query")
        assert code == 1

    def test_project_default_accepted(self, brain):
        run_cli("--project", "default", "test query")
        brain.switch_project.assert_called_with("default")


# ---------------------------------------------------------------------------
# 19.  main() --dry-run + query
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def _make_result(self, src="doc.pdf", sym=None):
        meta = {"source": src}
        if sym:
            meta["symbol_name"] = sym
        result = [{"id": "chunk1", "score": 0.95, "text": "some text here", "metadata": meta}]
        diag = MagicMock()
        diag.result_count = 1
        diag.to_dict.return_value = {"result_count": 1}
        return result, diag, {}

    def test_dry_run_calls_search_raw(self, brain):
        brain.search_raw.return_value = self._make_result()
        run_cli("--dry-run", "what is axon?")
        brain.search_raw.assert_called_once_with("what is axon?")

    def test_dry_run_does_not_call_query(self, brain):
        brain.search_raw.return_value = self._make_result()
        run_cli("--dry-run", "what is axon?")
        brain.query.assert_not_called()

    def test_dry_run_prints_dry_run_label(self, brain, capsys):
        brain.search_raw.return_value = self._make_result()
        run_cli("--dry-run", "what is axon?")
        assert "DRY RUN" in capsys.readouterr().out

    def test_dry_run_prints_ranked_chunks(self, brain, capsys):
        brain.search_raw.return_value = self._make_result(src="doc.pdf")
        run_cli("--dry-run", "what is axon?")
        assert "doc.pdf" in capsys.readouterr().out

    def test_dry_run_chunk_with_symbol_name(self, brain, capsys):
        brain.search_raw.return_value = self._make_result(src="code.py", sym="foo")
        run_cli("--dry-run", "what is foo?")
        out = capsys.readouterr().out
        assert "foo" in out
        assert "code.py" in out

    def test_dry_run_uses_file_path_fallback(self, brain, capsys):
        result = [
            {
                "id": "chunk1",
                "score": 0.8,
                "text": "some content",
                "metadata": {"file_path": "/path/to/file.py"},
            }
        ]
        diag = MagicMock()
        diag.result_count = 1
        diag.to_dict.return_value = {}
        brain.search_raw.return_value = (result, diag, {})
        run_cli("--dry-run", "test")
        assert "file.py" in capsys.readouterr().out

    def test_dry_run_uses_id_as_last_resort(self, brain, capsys):
        result = [{"id": "chunk-id-99", "score": 0.7, "text": "content", "metadata": {}}]
        diag = MagicMock()
        diag.result_count = 1
        diag.to_dict.return_value = {}
        brain.search_raw.return_value = (result, diag, {})
        run_cli("--dry-run", "test")
        assert "chunk-id-99" in capsys.readouterr().out

    def test_dry_run_prints_score(self, brain, capsys):
        brain.search_raw.return_value = self._make_result()
        run_cli("--dry-run", "test")
        assert "0.950" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 20.  RAG flags
# ---------------------------------------------------------------------------


class TestMainRagFlags:
    def test_hyde_enable(self, cfg):
        run_cli("--hyde", "test query")
        assert cfg.hyde is True

    def test_hyde_disable(self, cfg):
        run_cli("--no-hyde", "test query")
        assert cfg.hyde is False

    def test_rerank_enable(self, cfg):
        run_cli("--rerank", "test query")
        assert cfg.rerank is True

    def test_rerank_disable(self, cfg):
        run_cli("--no-rerank", "test query")
        assert cfg.rerank is False

    def test_hybrid_enable(self, cfg):
        run_cli("--hybrid", "test query")
        assert cfg.hybrid_search is True

    def test_hybrid_disable(self, cfg):
        run_cli("--no-hybrid", "test query")
        assert cfg.hybrid_search is False

    def test_no_discuss_disables_fallback(self, cfg):
        run_cli("--no-discuss", "test query")
        assert cfg.discussion_fallback is False

    def test_discuss_enables_fallback(self, cfg):
        run_cli("--discuss", "test query")
        assert cfg.discussion_fallback is True

    def test_multi_query_enable(self, cfg):
        run_cli("--multi-query", "test query")
        assert cfg.multi_query is True

    def test_multi_query_disable(self, cfg):
        run_cli("--no-multi-query", "test query")
        assert cfg.multi_query is False

    def test_step_back_enable(self, cfg):
        run_cli("--step-back", "test query")
        assert cfg.step_back is True

    def test_step_back_disable(self, cfg):
        run_cli("--no-step-back", "test query")
        assert cfg.step_back is False

    def test_decompose_enable(self, cfg):
        run_cli("--decompose", "test query")
        assert cfg.query_decompose is True

    def test_decompose_disable(self, cfg):
        run_cli("--no-decompose", "test query")
        assert cfg.query_decompose is False

    def test_compress_enable(self, cfg):
        run_cli("--compress", "test query")
        assert cfg.compress_context is True

    def test_compress_disable(self, cfg):
        run_cli("--no-compress", "test query")
        assert cfg.compress_context is False

    def test_cache_enable(self, cfg):
        run_cli("--cache", "test query")
        assert cfg.query_cache is True

    def test_cache_disable(self, cfg):
        run_cli("--no-cache", "test query")
        assert cfg.query_cache is False

    def test_raptor_enable(self, cfg):
        run_cli("--raptor", "test query")
        assert cfg.raptor is True

    def test_raptor_disable(self, cfg):
        run_cli("--no-raptor", "test query")
        assert cfg.raptor is False

    def test_graph_rag_enable(self, cfg):
        run_cli("--graph-rag", "test query")
        assert cfg.graph_rag is True

    def test_graph_rag_disable(self, cfg):
        run_cli("--no-graph-rag", "test query")
        assert cfg.graph_rag is False

    def test_cite_enable(self, cfg):
        run_cli("--cite", "test query")
        assert cfg.cite is True

    def test_cite_disable(self, cfg):
        run_cli("--no-cite", "test query")
        assert cfg.cite is False

    def test_code_graph_enable(self, cfg):
        run_cli("--code-graph", "test query")
        assert cfg.code_graph is True

    def test_code_graph_disable(self, cfg):
        run_cli("--no-code-graph", "test query")
        assert cfg.code_graph is False

    def test_code_graph_bridge_enable(self, cfg):
        run_cli("--code-graph-bridge", "test query")
        assert cfg.code_graph_bridge is True

    def test_code_graph_bridge_disable(self, cfg):
        run_cli("--no-code-graph-bridge", "test query")
        assert cfg.code_graph_bridge is False

    def test_threshold_sets_config(self, cfg):
        run_cli("--threshold", "0.4", "test query")
        assert cfg.similarity_threshold == pytest.approx(0.4)

    def test_reranker_model_sets_config(self, cfg):
        run_cli("--reranker-model", "BAAI/bge-reranker-v2-m3", "test query")
        assert cfg.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_no_dedup_disables_dedup(self, cfg):
        run_cli("--no-dedup", "test query")
        assert cfg.dedup_on_ingest is False

    def test_chunk_strategy_recursive(self, cfg):
        run_cli("--chunk-strategy", "recursive", "test query")
        assert cfg.chunk_strategy == "recursive"

    def test_chunk_strategy_semantic(self, cfg):
        run_cli("--chunk-strategy", "semantic", "test query")
        assert cfg.chunk_strategy == "semantic"

    def test_raptor_group_size(self, cfg):
        run_cli("--raptor-group-size", "10", "test query")
        assert cfg.raptor_chunk_group_size == 10

    def test_parent_chunk_size(self, cfg):
        run_cli("--parent-chunk-size", "512", "test query")
        assert cfg.parent_chunk_size == 512

    def test_search_enable(self, cfg):
        run_cli("--search", "test query")
        assert cfg.truth_grounding is True

    def test_search_disable(self, cfg):
        run_cli("--no-search", "test query")
        assert cfg.truth_grounding is False


# ---------------------------------------------------------------------------
# 21.  --embed flag
# ---------------------------------------------------------------------------


class TestMainEmbed:
    def test_embed_plain_sets_model(self, cfg):
        run_cli("--embed", "all-MiniLM-L6-v2", "test query")
        assert cfg.embedding_model == "all-MiniLM-L6-v2"

    def test_embed_slash_known_provider_ollama(self, cfg):
        run_cli("--embed", "ollama/nomic-embed-text", "test query")
        assert cfg.embedding_provider == "ollama"
        assert cfg.embedding_model == "nomic-embed-text"

    def test_embed_slash_known_provider_openai(self, cfg):
        run_cli("--embed", "openai/text-embedding-3-small", "test query")
        assert cfg.embedding_provider == "openai"
        assert cfg.embedding_model == "text-embedding-3-small"

    def test_embed_slash_unknown_provider(self, cfg):
        run_cli("--embed", "custom/my-embed-model", "test query")
        # unknown prefix → just sets embedding_model to full string
        assert cfg.embedding_model == "custom/my-embed-model"

    def test_embed_fastembed_provider(self, cfg):
        run_cli("--embed", "fastembed/BAAI/bge-small-en", "test query")
        assert cfg.embedding_provider == "fastembed"
        assert cfg.embedding_model == "BAAI/bge-small-en"


# ---------------------------------------------------------------------------
# 22.  Edge cases and misc integration
# ---------------------------------------------------------------------------


class TestMainEdgeCases:
    def test_config_path_passed_to_load(self):
        with patch("axon.config.AxonConfig.load") as _load:
            _load.return_value = MagicMock(
                llm_provider="ollama", llm_model="gemma:2b", vllm_base_url=""
            )
            run_cli("--config", "/tmp/custom_config.yaml", "test query")
        _load.assert_called_with("/tmp/custom_config.yaml")

    def test_quiet_flag_accepted(self):
        code = run_cli("--quiet")
        assert code == 0 or code is None

    def test_short_quiet_flag_accepted(self):
        code = run_cli("-q")
        assert code == 0 or code is None

    def test_list_after_ingest(self, brain, capsys):
        brain.list_documents.return_value = [{"source": "ingested.txt", "chunks": 2}]
        with patch("os.path.isdir", return_value=True), patch("asyncio.run"):
            run_cli("--ingest", "/some/dir", "--list")
        assert "ingested.txt" in capsys.readouterr().out

    def test_project_list_with_nested_children(self, brain, capsys):
        brain._active_project = "parent"
        child = {"name": "parent/child", "created_at": "2024-03-01T00:00:00", "children": []}
        projects = [{"name": "parent", "created_at": "2024-01-01T00:00:00", "children": [child]}]
        with patch("axon.projects.list_projects", return_value=projects):
            run_cli("--project-list")
        out = capsys.readouterr().out
        assert "parent" in out
        assert "child" in out

    def test_ingest_unknown_extension_not_ingested(self, brain, tmp_path):
        test_file = tmp_path / "file.xyz"
        test_file.write_text("data")
        loader_mock = MagicMock()
        loader_mock.loaders = {}  # no loader for .xyz
        with (
            patch("os.path.isdir", return_value=False),
            patch("axon.loaders.DirectoryLoader", return_value=loader_mock),
        ):
            run_cli("--ingest", str(test_file))
        brain.ingest.assert_not_called()

    def test_query_multiword_passed_as_single_arg(self, brain):
        brain.query.return_value = "ok"
        run_cli("how does axon work?")
        brain.query.assert_called_once_with("how does axon work?")

    def test_project_delete_returns_early_no_query(self, brain):
        brain._active_project = "other"
        with patch("axon.projects.delete_project"):
            run_cli("--project-delete", "myproject")
        brain.query.assert_not_called()
