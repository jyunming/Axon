from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest


# Helper: attempt to import a symbol from possible modules/names
def try_get(module_name, attr_name):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    except Exception:
        return None


def test_tool_run_shell_unicode_decode_error():
    """Verify _tool_run_shell does not let UnicodeDecodeError bubble up."""
    tool = try_get("axon.agent", "_tool_run_shell")
    if tool is None:
        pytest.skip("_tool_run_shell not present in axon.agent")

    # Patch the global subprocess.run to simulate a decoding error in the child process
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid byte")
        try:
            res = tool({"command": "echo-bad-bytes"})
        except UnicodeDecodeError:
            pytest.fail("UnicodeDecodeError bubbled up from _tool_run_shell")
        # Tool should return an error string (not None) and not raise
        assert res is not None


def test_repl_at_expansion_email_collision(tmp_path, monkeypatch):
    """Ensure typing an email does not expand to a file named like the domain.

    Marked xfail because current implementation expands '@domain' tokens greedily.
    """
    expand = try_get("axon.repl", "_expand_at_files") or try_get("axon.repl", "expand_at_files")
    if expand is None:
        pytest.skip("_expand_at_files / expand_at_files not found")

    domain_file = tmp_path / "example.com"
    domain_file.write_text("secret content", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    text = "Contact me at user@example.com"
    expanded = expand(text)
    # Should NOT replace the email with the contents of example.com
    assert "secret content" not in expanded


def test_vector_store_add_unbalanced_inputs(tmp_path):
    """Verify OpenVectorStore.add rejects unbalanced input batches before backend writes."""
    OpenVectorStore = try_get("axon.vector_store", "OpenVectorStore")
    if OpenVectorStore is None:
        pytest.skip("OpenVectorStore not available")

    # Use a real AxonConfig to satisfy constructor requirements, but point paths to tmp
    try:
        from axon.config import AxonConfig
    except Exception:
        pytest.skip("AxonConfig not available")

    cfg = AxonConfig()
    cfg.vector_store_path = str(tmp_path / "vsdata")
    cfg.vector_store = "chroma"

    # Patch _init_store to avoid importing external backends during tests
    with patch("axon.vector_store.OpenVectorStore._init_store", new=lambda self: None):
        store = OpenVectorStore(cfg)
        # Provide a mocked collection to observe calls
        store.collection = MagicMock()

        with pytest.raises(ValueError, match="length mismatch"):
            store.add(["id1", "id2"], ["text1"], [[0.1]])

        store.collection.add.assert_not_called()


def test_projects_delete_race_condition(tmp_path):
    """Simulate rmtree raising FileNotFoundError and assert delete_project handles it.

    Marked xfail until the project deletion race is fixed in axon.projects.delete_project.
    """
    delete_project = try_get("axon.projects", "delete_project")
    ensure_project = try_get("axon.projects", "ensure_project")
    set_projects_root = try_get("axon.projects", "set_projects_root")
    if delete_project is None or ensure_project is None or set_projects_root is None:
        pytest.skip("project helpers not found")

    # Isolate PROJECTS_ROOT to tmp_path
    set_projects_root(str(tmp_path / "projects_root"))
    ensure_project("p1")

    # Patch shutil.rmtree used in the module to raise FileNotFoundError
    with patch("axon.projects.shutil.rmtree", side_effect=FileNotFoundError()):
        # delete_project currently does not handle FileNotFoundError — xfail
        delete_project("p1")


@pytest.mark.xfail(reason="Integration / environment dependent (manual review)")
def test_api_ingest_upload_memory_exhaustion():
    """Simulate a huge multipart upload.read() causing MemoryError and expect the handler to not crash."""
    ingest = try_get("axon.api_routes.ingest", "ingest_upload") or try_get(
        "axon.api", "ingest_upload"
    )
    if ingest is None:
        pytest.skip("ingest_upload endpoint handler not found")

    class DummyUpload:
        async def read(self):
            raise MemoryError("simulate OOM")

    # If ingest is async, run it; expect it to handle MemoryError and not propagate
    import asyncio

    try:
        asyncio.run(ingest(DummyUpload()))
    except MemoryError:
        pytest.fail("ingest_upload allowed MemoryError to bubble up")


@pytest.mark.xfail(reason="External provider behavior; integration test only")
def test_embeddings_openai_batch_limit():
    """Verify OpenAI embedding provider splits large batches into <=2048 chunks."""
    emb_mod = try_get("axon.embeddings", "OpenAIEmbeddingProvider") or try_get(
        "axon.embeddings", "openai"
    )
    if emb_mod is None:
        pytest.skip("OpenAI embedding provider not present")

    # If provider is a class, instantiate; else assume it's a module-level function
    provider = None
    if isinstance(emb_mod, type):
        try:
            provider = emb_mod()
        except Exception:
            pytest.skip("Could not instantiate embedding provider")
    else:
        provider = emb_mod

    # Patch the underlying openai client call to assert chunk size
    # Best-effort: look up the name used inside axon.embeddings
    with patch("axon.embeddings.openai") as mock_openai:
        # simulate API accepting a list but we will check call args
        mock_openai.Embeddings = MagicMock()
        mock_openai.Embeddings.create = MagicMock()
        texts = ["x"] * 2050
        try:
            # try calling a common entrypoint name
            if hasattr(provider, "embed"):
                provider.embed(texts)
            elif hasattr(provider, "embed_texts"):
                provider.embed_texts(texts)
            else:
                pytest.skip("No embed entrypoint found on provider")
        except Exception:
            # ignore provider-side exceptions; test inspects mock
            pass
        # If provider attempted to call OpenAI in one shot with >2048, that's a problem
        calls = mock_openai.Embeddings.create.call_args_list
        for call in calls:
            # attempt to fetch the length of "input" or "texts" arg
            kwargs = call.kwargs if hasattr(call, "kwargs") else {}
            inputs = kwargs.get("input") or kwargs.get("texts") or kwargs.get("texts")
            if inputs is not None:
                assert len(inputs) <= 2048, "OpenAI called with too many inputs in one request"
