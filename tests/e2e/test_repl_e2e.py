from __future__ import annotations

from collections.abc import Iterable

import pytest

import axon.main as main_module
import axon.repl as repl_module
import axon.sessions as sessions_module

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def _run_repl_transcript(monkeypatch, home_dir, commands: Iterable[str]):
    prompt_toolkit = pytest.importorskip("prompt_toolkit")
    scripted = iter(commands)

    class _FakePromptSession:
        def __init__(self, *args, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            return next(scripted)

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("USERPROFILE", str(home_dir))
    monkeypatch.setattr(repl_module, "_draw_header", lambda *args, **kwargs: None)
    monkeypatch.setattr(sessions_module, "_SESSIONS_DIR", str(home_dir / ".axon" / "sessions"))
    monkeypatch.setattr(prompt_toolkit, "PromptSession", _FakePromptSession)


def test_repl_transcript_ingest_query_list_and_sessions(
    make_brain, monkeypatch, sample_docs_dir, tmp_path, capsys
):
    brain = make_brain()
    monkeypatch.setattr(brain, "should_recommend_project", lambda: False)
    _run_repl_transcript(
        monkeypatch,
        tmp_path,
        [
            f"/ingest {sample_docs_dir}",
            "What does GraphRAG link together?",
            "/list",
            "/sessions",
            "/quit",
        ],
    )

    main_module._interactive_repl(brain, stream=False, quiet=True)
    output = capsys.readouterr().out
    sessions = main_module._list_sessions(project="default")

    assert "Done — 1 ingested, 0 skipped." in output
    assert str(sample_docs_dir / "notes.md") in output
    assert len(sessions) == 1
    assert sessions[0]["history"][0]["content"] == "What does GraphRAG link together?"
    assistant_turns = [turn for turn in sessions[0]["history"] if turn["role"] == "assistant"]
    assert assistant_turns and assistant_turns[0]["content"].strip()
    assert "fake-llm" in output


def test_repl_transcript_project_and_runtime_controls(make_brain, monkeypatch, tmp_path, capsys):
    brain = make_brain()
    _run_repl_transcript(
        monkeypatch,
        tmp_path,
        [
            "/project new research Research notes",
            "/project list",
            "/project switch default",
            "/rag topk 3",
            "/rag hybrid",
            "/llm temperature 0.2",
            "/clear",
            "/quit",
        ],
    )

    main_module._interactive_repl(brain, stream=False, quiet=True)
    output = capsys.readouterr().out

    assert "Created and switched to project 'research'" in output
    assert "research" in output
    assert "Switched to project 'default'" in output
    assert "top-k set to 3" in output
    assert "Hybrid search OFF" in output
    assert "Temperature set to 0.2" in output
    assert "Chat history cleared." in output
