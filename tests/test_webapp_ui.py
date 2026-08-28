import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("streamlit", reason="streamlit not installed — run: pip install axon-rag[ui]")
from streamlit.testing.v1 import AppTest


class TestWebappUI:
    @pytest.fixture
    def app(self):
        # Add src to sys.path
        src_path = str(Path(__file__).parent.parent / "src")
        if src_path not in sys.path:
            sys.path.append(src_path)

        # Mocking things before the app loads
        # We must use a context manager that stays active during at.run()
        # but AppTest runs in a separate thread/process sometimes.
        # Actually AppTest.from_file loads the module.

        with (
            patch("axon.webapp.AxonBrain") as mock_brain_cls,
            patch("axon.webapp.load_sessions", return_value={}),
            patch("axon.webapp.list_projects", return_value=[{"name": "default"}]),
            patch("axon.webapp.get_active_project", return_value="default"),
        ):
            mock_brain = mock_brain_cls.return_value
            mock_brain.config = MagicMock()
            mock_brain.config.embedding_provider = "sentence_transformers"
            mock_brain.config.embedding_model = "all-MiniLM-L6-v2"
            mock_brain.config.llm_provider = "openai"
            mock_brain.config.llm_model = "gpt-4o"
            mock_brain.config.llm_temperature = 0.7
            mock_brain.config.top_k = 8
            mock_brain.config.similarity_threshold = 0.5
            mock_brain.config.hybrid_search = False
            mock_brain.config.discussion_fallback = True
            mock_brain.config.rerank = False
            mock_brain.config.truth_grounding = False
            mock_brain.config.multi_query = False
            mock_brain.config.hyde = False
            mock_brain.config.step_back = False
            mock_brain.config.query_decompose = False
            mock_brain.config.compress_context = False
            mock_brain.config.raptor = False
            mock_brain.config.graph_rag = False

            mock_brain.list_documents.return_value = []

            # Yield tokens and then a sources dict, as webapp.py expects
            def mock_query_stream(query, chat_history=None):
                yield "Mocked "
                yield "response"
                yield {
                    "type": "sources",
                    "sources": [{"id": "doc1", "text": "context", "score": 0.9, "metadata": {}}],
                }

            mock_brain.query_stream.side_effect = mock_query_stream

            at = AppTest.from_file("src/axon/webapp.py", default_timeout=60)
            # We need to keep the patches active?
            # AppTest.run() executes the script.
            # If it's in-process, it should work.
            yield at

    def test_ui_initialization(self, app):
        at = app.run()
        assert not at.exception
        # Title check
        assert any("Axon" in str(m.value) for m in at.markdown)

    @pytest.mark.skip(
        reason="st.spinner + streaming generator causes AppTest timeout; "
        "streaming behaviour is covered by tests/test_streaming.py"
    )
    def test_ui_query_interaction(self, app):
        at = app.run()
        # The chat_input is at index 0
        at.chat_input[0].set_value("Who are you?").run()

        # Check if query was called (it should be if it didn't crash)
        # We can check session state messages
        assert len(at.session_state.sessions[at.session_state.current_session_id]["messages"]) >= 1

    def test_ui_sidebar_toggles(self, app):
        at = app.run()

        # Target by label "Hybrid search"
        # at.checkbox returns a list of checkboxes, we can filter by label
        hybrid_cb = [c for c in at.checkbox if c.label == "Hybrid search"][0]
        hybrid_cb.set_value(True).run()

        # Verify it updated the config in session state
        assert at.session_state.brain.config.hybrid_search is True

    def test_ui_shows_deprecation_banner(self, app):
        """The Streamlit UI must tell users it is deprecated and where to go."""
        at = app.run()
        assert not at.exception
        banners = [str(w.value) for w in at.warning]
        assert any("deprecated" in b.lower() for b in banners), banners


class TestCheckIngestDirDenied:
    """_check_ingest_dir_denied() must enforce the same system-path
    blocklist api_schemas._validate_ingest_path() applies to the REST
    /ingest route — previously this Streamlit-only path checked
    RAG_INGEST_BASE containment but had no blocklist at all, so a broad or
    unset RAG_INGEST_BASE gave no protection against ingesting a system
    directory. A pure function (no Streamlit dependency), so tested with a
    plain import rather than the AppTest harness used above."""

    def test_blocked_system_path_denied_even_with_permissive_base(self, monkeypatch, tmp_path):
        from axon.webapp import _check_ingest_dir_denied

        # RAG_INGEST_BASE unset defaults to "." — permissive enough that the
        # base-containment check alone would not have caught a system path.
        monkeypatch.delenv("RAG_INGEST_BASE", raising=False)
        monkeypatch.chdir(tmp_path)
        blocked = "C:/Windows" if os.name == "nt" else "/etc"
        denial = _check_ingest_dir_denied(blocked)
        assert denial is not None
        assert "blocked system path" in denial

    def test_path_outside_allowed_base_denied(self, monkeypatch, tmp_path):
        from axon.webapp import _check_ingest_dir_denied

        allowed = tmp_path / "allowed"
        outside = tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        monkeypatch.setenv("RAG_INGEST_BASE", str(allowed))
        denial = _check_ingest_dir_denied(str(outside))
        assert denial is not None
        assert "outside allowed base" in denial

    def test_path_inside_allowed_base_permitted(self, monkeypatch, tmp_path):
        from axon.webapp import _check_ingest_dir_denied

        allowed = tmp_path / "allowed"
        nested = allowed / "sub"
        nested.mkdir(parents=True)
        monkeypatch.setenv("RAG_INGEST_BASE", str(allowed))
        assert _check_ingest_dir_denied(str(nested)) is None
