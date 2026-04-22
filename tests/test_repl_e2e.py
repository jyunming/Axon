import io
import os
from unittest.mock import patch

import pytest

import axon.repl as _repl_mod
from axon.cli import main


class TestReplE2E:
    @pytest.fixture
    def mock_env(self, tmp_path):
        env = os.environ.copy()
        env["AXON_HOME"] = str(tmp_path / ".axon")
        env["OPENAI_API_KEY"] = "sk-mock"
        env["GITHUB_COPILOT_PAT"] = "mock-pat"
        env["AXON_QUIET"] = "1"
        with patch.dict(os.environ, env, clear=True):
            yield env

    def test_repl_basic_flow_in_process(self, mock_env, tmp_path):
        """Test a basic REPL session by mocking input/output in-process."""
        _orig_repl = _repl_mod._interactive_repl

        def _scripted_repl(brain, **kwargs):
            kwargs["_scripted_inputs"] = ["/help", "/exit"]
            return _orig_repl(brain, **kwargs)

        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ), patch("axon.main.OpenLLM") as mock_llm_cls, patch("axon.main.OpenReranker"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_project"
        ), patch.object(
            _repl_mod, "_interactive_repl", side_effect=_scripted_repl
        ):
            mock_llm = mock_llm_cls.return_value
            mock_llm.complete.return_value = "Mocked response"

            output_buffer = io.StringIO()

            with patch("sys.argv", ["axon"]):
                with patch("sys.stdout", output_buffer):
                    with patch(
                        "builtins.print",
                        side_effect=lambda *args, **kwargs: output_buffer.write(
                            " ".join(map(str, args)) + "\n"
                        ),
                    ):
                        with patch("os._exit"):
                            with patch("sys.stdin.isatty", return_value=True):
                                main()

            output = output_buffer.getvalue()

            assert "/clear" in output or "/compact" in output or "help" in output
            assert "Bye" in output
