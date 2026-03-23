import io
import os
from unittest.mock import patch

import pytest

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
        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ), patch("axon.main.OpenLLM") as mock_llm_cls, patch("axon.main.OpenReranker"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_namespace"
        ), patch(
            "prompt_toolkit.formatted_text.ANSI", side_effect=lambda x: x
        ), patch(
            "prompt_toolkit.PromptSession"
        ) as mock_ps_cls:
            mock_llm = mock_llm_cls.return_value
            mock_llm.complete.return_value = "Mocked response"

            mock_ps = mock_ps_cls.return_value
            # Inputs: /help, then /exit
            mock_ps.prompt.side_effect = ["/help", "/exit"]

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
            # print(output) # For debugging if needed

            assert "/clear" in output or "/compact" in output or "help" in output
            assert "Bye" in output
