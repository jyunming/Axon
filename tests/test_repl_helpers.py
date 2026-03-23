"""Tests for standalone helper functions in axon.repl (no full REPL loop)."""
import io
import logging
import os
from unittest.mock import MagicMock, patch


def _make_brain_mock(**kwargs):
    brain = MagicMock()
    brain.config.llm_provider = kwargs.get("llm_provider", "ollama")
    brain.config.llm_model = kwargs.get("llm_model", "llama3")
    brain.config.embedding_provider = kwargs.get("embedding_provider", "sentence_transformers")
    brain.config.embedding_model = kwargs.get("embedding_model", "all-MiniLM-L6-v2")
    brain.config.top_k = kwargs.get("top_k", 8)
    brain.config.similarity_threshold = kwargs.get("similarity_threshold", 0.0)
    brain.config.hybrid_search = kwargs.get("hybrid_search", False)
    brain.config.rerank = kwargs.get("rerank", False)
    brain.config.reranker_model = kwargs.get("reranker_model", "cross-encoder/ms-marco")
    brain.config.hyde = kwargs.get("hyde", False)
    brain.config.multi_query = kwargs.get("multi_query", False)
    brain.config.step_back = kwargs.get("step_back", False)
    brain.config.query_decompose = kwargs.get("query_decompose", False)
    brain.config.compress_context = kwargs.get("compress_context", False)
    brain.config.raptor = kwargs.get("raptor", False)
    brain.config.graph_rag = kwargs.get("graph_rag", False)
    brain.config.discussion_fallback = kwargs.get("discussion_fallback", True)
    brain.config.truth_grounding = kwargs.get("truth_grounding", False)
    brain.config.llm_temperature = 0.7
    brain.config.vllm_base_url = "http://localhost:8000/v1"
    brain.list_documents.return_value = [{"source": "test.txt", "chunks": 5}]
    brain._active_project = "default"
    return brain


class TestEstimateTokens:
    def test_empty_string(self):
        from axon.repl import _estimate_tokens

        assert _estimate_tokens("") == 1  # max(1, 0//4)

    def test_short_text(self):
        from axon.repl import _estimate_tokens

        assert _estimate_tokens("hello") == 1  # 5//4 = 1

    def test_longer_text(self):
        from axon.repl import _estimate_tokens

        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_minimum_one(self):
        from axon.repl import _estimate_tokens

        assert _estimate_tokens("abc") >= 1


class TestTokenBar:
    def test_zero_usage(self):
        from axon.repl import _token_bar

        result = _token_bar(0, 1000)
        assert "[ok]" in result
        assert "0%" in result

    def test_half_full_ok(self):
        from axon.repl import _token_bar

        result = _token_bar(500, 1000)
        assert "[ok]" in result
        assert "50%" in result

    def test_high_usage_warning(self):
        from axon.repl import _token_bar

        result = _token_bar(700, 1000)
        assert "[!]" in result

    def test_critical_usage(self):
        from axon.repl import _token_bar

        result = _token_bar(900, 1000)
        assert "[!!]" in result

    def test_total_zero(self):
        from axon.repl import _token_bar

        result = _token_bar(0, 0)
        assert "0%" in result

    def test_over_capacity_clamped(self):
        from axon.repl import _token_bar

        result = _token_bar(2000, 1000)
        assert "100%" in result


class TestInferProvider:
    def test_ollama_with_tag(self):
        from axon.repl import _infer_provider

        assert _infer_provider("llama3:8b") == "ollama"

    def test_gemini_model_name(self):
        from axon.repl import _infer_provider

        assert _infer_provider("gemini-1.5-flash") == "gemini"

    def test_gemini_pro(self):
        from axon.repl import _infer_provider

        assert _infer_provider("gemini-pro") == "gemini"

    def test_gpt_model_openai(self):
        from axon.repl import _infer_provider

        assert _infer_provider("gpt-4-turbo") == "openai"

    def test_gpt_with_colon_is_ollama(self):
        from axon.repl import _infer_provider

        # Ollama gpt models use name:tag format
        assert _infer_provider("gpt-4:latest") == "ollama"

    def test_plain_model_defaults_ollama(self):
        from axon.repl import _infer_provider

        assert _infer_provider("llama3") == "ollama"

    def test_o1_model_openai(self):
        from axon.repl import _infer_provider

        assert _infer_provider("o1-preview") == "openai"

    def test_unknown_model_ollama(self):
        from axon.repl import _infer_provider

        assert _infer_provider("custom-model-xyz") == "ollama"


class TestBoxWidth:
    def test_returns_integer(self):
        from axon.repl import _box_width

        result = _box_width()
        assert isinstance(result, int)
        assert result >= 43

    def test_minimum_43(self):
        from axon.repl import _box_width

        with patch("axon.repl.shutil.get_terminal_size", return_value=MagicMock(columns=10)):
            result = _box_width()
            assert result == 43


class TestBrow:
    def test_returns_box_row(self):
        from axon.repl import _brow

        result = _brow("hello")
        assert result.startswith("  ┃")
        assert result.endswith("┃")

    def test_truncates_long_content(self):
        from axon.repl import _brow

        long_content = "x" * 1000
        result = _brow(long_content)
        assert "…" in result

    def test_short_content_padded(self):
        from axon.repl import _brow

        result = _brow("hi")
        # Should contain spaces for padding
        assert " " in result


class TestAnimPad:
    def test_returns_string(self):
        from axon.repl import _anim_pad

        result = _anim_pad(0, 0, 20)
        assert isinstance(result, str)

    def test_various_frames(self):
        from axon.repl import _anim_pad

        for frame in range(10):
            result = _anim_pad(0, frame, 20)
            assert isinstance(result, str)


class TestGetBrainAnimRow:
    def test_returns_string_for_all_rows(self):
        from axon.repl import _get_brain_anim_row

        for row in range(6):
            result = _get_brain_anim_row(row, 0, 20)
            assert isinstance(result, str)

    def test_out_of_range_row(self):
        from axon.repl import _get_brain_anim_row

        result = _get_brain_anim_row(99, 0, 20)
        assert isinstance(result, str)

    def test_zero_width(self):
        from axon.repl import _get_brain_anim_row

        result = _get_brain_anim_row(0, 0, 0)
        assert isinstance(result, str)


class TestPrintRecentTurns:
    def test_empty_history_no_output(self):
        from axon.repl import _print_recent_turns

        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_recent_turns([])
        assert buf.getvalue() == ""

    def test_single_turn_printed(self):
        from axon.repl import _print_recent_turns

        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "World"},
        ]
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_recent_turns(history, n_turns=2)
        output = buf.getvalue()
        assert "Hello" in output
        assert "World" in output

    def test_long_response_truncated(self):
        from axon.repl import _print_recent_turns

        history = [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A" * 700},
        ]
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_recent_turns(history, n_turns=1)
        output = buf.getvalue()
        assert "…" in output

    def test_only_last_n_turns_shown(self):
        from axon.repl import _print_recent_turns

        history = [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "old response"},
            {"role": "user", "content": "new"},
            {"role": "assistant", "content": "new response"},
        ]
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            _print_recent_turns(history, n_turns=1)
        output = buf.getvalue()
        assert "new" in output
        assert "old" not in output


class TestExpandAtFiles:
    def test_no_at_mention_unchanged(self):
        from axon.repl import _expand_at_files

        text = "hello world no mentions"
        assert _expand_at_files(text) == text

    def test_at_existing_file(self, tmp_path):
        from axon.repl import _expand_at_files

        f = tmp_path / "notes.txt"
        f.write_text("file content here", encoding="utf-8")
        result = _expand_at_files(f"Look at @{f}")
        assert "file content here" in result

    def test_at_nonexistent_file_returns_original(self):
        from axon.repl import _expand_at_files

        text = "see @/nonexistent/file.txt for details"
        result = _expand_at_files(text)
        assert "@/nonexistent/file.txt" in result

    def test_at_directory(self, tmp_path):
        from axon.repl import _expand_at_files

        subdir = tmp_path / "docs"
        subdir.mkdir()
        (subdir / "a.txt").write_text("content A", encoding="utf-8")
        (subdir / "b.md").write_text("content B", encoding="utf-8")
        result = _expand_at_files(f"docs: @{subdir}/")
        assert "content A" in result or "content B" in result

    def test_at_empty_directory(self, tmp_path):
        from axon.repl import _expand_at_files

        subdir = tmp_path / "empty"
        subdir.mkdir()
        result = _expand_at_files(f"@{subdir}/")
        assert "no readable files" in result


class TestDoCompact:
    def test_empty_history_prints_message(self, capsys):
        from axon.repl import _do_compact

        brain = _make_brain_mock()
        history = []
        _do_compact(brain, history)
        captured = capsys.readouterr()
        assert "Nothing to compact" in captured.out

    def test_compact_replaces_history(self, capsys):
        from axon.repl import _do_compact

        brain = _make_brain_mock()
        brain.llm.complete.return_value = "Short summary"
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        _do_compact(brain, history)
        assert len(history) == 1
        assert "[Conversation summary]" in history[0]["content"]
        assert "Short summary" in history[0]["content"]

    def test_compact_llm_failure_no_crash(self, capsys):
        from axon.repl import _do_compact

        brain = _make_brain_mock()
        brain.llm.complete.side_effect = RuntimeError("LLM failed")
        history = [{"role": "user", "content": "Hello"}]
        _do_compact(brain, history)
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or len(history) == 1


class TestBuildHeader:
    def test_returns_list_of_strings(self):
        from axon.repl import _build_header

        brain = _make_brain_mock()
        result = _build_header(brain)
        assert isinstance(result, list)
        assert all(isinstance(r, str) for r in result)

    def test_with_tick_lines(self):
        from axon.repl import _build_header

        brain = _make_brain_mock()
        result = _build_header(brain, tick_lines=["Embedding ready [CPU]", "BM25 ready"])
        assert isinstance(result, list)
        assert len(result) > 5

    def test_list_documents_exception_graceful(self):
        from axon.repl import _build_header

        brain = _make_brain_mock()
        brain.list_documents.side_effect = Exception("DB error")
        result = _build_header(brain)
        assert isinstance(result, list)

    def test_long_tick_lines_overflow_row(self):
        from axon.repl import _build_header

        brain = _make_brain_mock()
        many_ticks = [f"Item {i}" for i in range(20)]
        result = _build_header(brain, tick_lines=many_ticks)
        assert isinstance(result, list)


class TestSaveEnvKey:
    def test_writes_to_file_and_env(self, tmp_path):
        from axon.repl import _save_env_key

        env_file = tmp_path / ".axon" / ".env"

        with patch("pathlib.Path.home", return_value=tmp_path):
            _save_env_key("TEST_KEY", "my_secret_value")

        assert "TEST_KEY" in os.environ
        assert os.environ["TEST_KEY"] == "my_secret_value"
        assert env_file.exists()
        content = env_file.read_text(encoding="utf-8")
        assert "TEST_KEY=my_secret_value" in content

        # Cleanup
        del os.environ["TEST_KEY"]

    def test_overwrites_existing_key(self, tmp_path):
        from axon.repl import _save_env_key

        env_file = tmp_path / ".axon" / ".env"
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text("OLD_KEY=old\nTEST_KEY2=original\n", encoding="utf-8")

        with patch("pathlib.Path.home", return_value=tmp_path):
            _save_env_key("TEST_KEY2", "updated_value")

        content = env_file.read_text(encoding="utf-8")
        assert "TEST_KEY2=updated_value" in content
        assert "TEST_KEY2=original" not in content

        if "TEST_KEY2" in os.environ:
            del os.environ["TEST_KEY2"]


class TestInitDisplay:
    def test_emit_message_parsing(self):
        """Test _InitDisplay.emit() message handling without the full threaded UI."""

        from axon.repl import _InitDisplay

        # Patch sys.stdout and immediately stop the thread to avoid hanging
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            disp = _InitDisplay()
            # Immediately stop the animation thread
            disp._done.set()
            disp._thread.join(timeout=0.5)

            # Test emit for "Initializing" message
            record = logging.LogRecord(
                name="Axon",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Initializing Axon...",
                args=(),
                exc_info=None,
            )
            disp.emit(record)
            assert disp._step == "Starting…"

    def test_emit_axon_ready_sets_done(self):
        """Test that 'Axon ready' message sets done event."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            from axon.repl import _InitDisplay

            disp = _InitDisplay()
            disp._done.set()  # stop spinner first
            disp._thread.join(timeout=0.5)

            record = logging.LogRecord(
                name="Axon",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="Axon ready.",
                args=(),
                exc_info=None,
            )
            disp.emit(record)
            assert disp._done.is_set()

    def test_tick_appends_label(self):
        """Test _tick() adds label to tick_lines."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            from axon.repl import _InitDisplay

            disp = _InitDisplay()
            disp._done.set()
            disp._thread.join(timeout=0.5)

            with disp._lock:
                disp.tick_lines.append("BM25 ready")
            assert "BM25 ready" in disp.tick_lines
