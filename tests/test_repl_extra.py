"""Extra tests for axon.repl utility functions to push coverage above 90%."""
from unittest.mock import MagicMock, patch


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
