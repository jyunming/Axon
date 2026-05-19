from unittest.mock import patch

from axon.config_wizard import run_wizard


def test_wizard_abort():
    # Sequence: Mode="1", LLM Skip="n", Embedding Skip="n", Vector Store Skip="n", Save="n"
    with patch("builtins.input", side_effect=["1", "n", "n", "n", "n"]):
        res = run_wizard()
        assert res == {}


def test_wizard_full_mode_tqdb_pick_does_not_crash():
    """Regression: ``_pick`` previously called ``.lower()`` on bool/int choices
    (``[False, True]``, ``[4, 8]``), so reaching the turboquantdb knobs in
    ``full`` mode raised ``AttributeError: 'bool' object has no attribute
    'lower'``. The fix preserves defaults without crashing.
    """
    inputs = [
        "3",  # mode: full
        "n",  # skip LLM
        "n",  # skip Embedding
        "y",  # enter Vector Store section (full-mode tqdb knobs live here)
        "1",  # provider: turboquantdb (default — no change)
        "",  # vector_store.path: keep default
        "",  # qdrant.url: keep
        "",  # qdrant.api_key: keep
        # The tqdb knobs are passed bool/int choices in source; before the
        # fix they crashed here. Press Enter to keep each default value.
        "",  # tqdb_bits  (choices = [4, 8])
        "",  # tqdb_fast_mode  (choices = [False, True])
        "",  # tqdb_rerank  (choices = [True, False])
        # Sections 4–16 (full-mode only) are all skipped to keep the test focused.
        "n",  # skip Chunking
        "n",  # skip Retrieval
        "n",  # skip Sentence Window & CRAG-Lite
        "n",  # skip Reranking
        "n",  # skip Query Transformations
        "n",  # skip RAPTOR
        "n",  # skip GraphRAG
        "n",  # skip Code Graph
        "n",  # skip Output
        "n",  # skip Performance
        "n",  # skip REPL
        "n",  # skip Offline
        "n",  # skip Store & Paths
    ]
    with patch("builtins.input", side_effect=inputs):
        res = run_wizard()
    # No changes were entered — every section was either skipped or kept its
    # default value. The point of the test is that this completes without
    # raising AttributeError.
    assert "tqdb_bits" not in res
    assert "tqdb_fast_mode" not in res
    assert "tqdb_rerank" not in res


def test_wizard_full_mode_tqdb_records_typed_values():
    """When the user picks a non-default tqdb choice the wizard must store
    the original Python type (int / bool) — not the stringified form — so
    downstream ``AxonConfig`` writes don't break dataclass invariants.
    """
    # With no brain, the wizard's `_pick` defaults are tqdb_bits=8,
    # tqdb_fast_mode=False, tqdb_rerank=True. Pick the OTHER option for
    # each so the changes are non-default and visible in `res`.
    inputs = [
        "3",  # mode: full
        "n",  # skip LLM
        "n",  # skip Embedding
        "y",  # enter Vector Store
        "1",  # provider: turboquantdb
        "",  # path
        "",  # qdrant.url
        "",  # qdrant.api_key
        "1",  # tqdb_bits: index 1 → 4 (wizard default 8 is at index 2)
        "2",  # tqdb_fast_mode: index 2 → True (wizard default False at index 1)
        "2",  # tqdb_rerank: index 2 → False (wizard default True at index 1)
        "n",  # skip Chunking ... onward
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "n",
        "y",  # save the changes
    ]
    with patch("builtins.input", side_effect=inputs), patch("axon.config.AxonConfig.save"):
        res = run_wizard()
    assert res.get("tqdb_bits") == 4
    assert isinstance(res["tqdb_bits"], int) and not isinstance(res["tqdb_bits"], bool)
    assert res.get("tqdb_fast_mode") is True
    assert isinstance(res["tqdb_fast_mode"], bool)
    assert res.get("tqdb_rerank") is False
    assert isinstance(res["tqdb_rerank"], bool)


def test_wizard_quick_mode_minimal():
    # To ensure it shows up in 'res', we must pick something DIFFERENT than default.
    # Default llm_model is 'llama3.1:8b'

    inputs = [
        "1",  # mode quick
        "y",  # config LLM
        "1",  # ollama
        "llama3.1:70b",  # DIFFERENT MODEL
        "",  # base url
        "",
        "",
        "",
        "",  # keys
        "y",  # config Embedding
        "1",  # sentence_transformers
        "BAAI/bge-small-en-v1.5",  # DIFFERENT EMBEDDING
        "",  # path
        "y",  # config Vector Store
        "3",  # qdrant (DIFFERENT — turboquantdb=1, lancedb=2, qdrant=3)
        "",  # path
        "http://localhost:6333",  # url
        "",  # key
        "y",  # SAVE
    ]

    with patch("builtins.input", side_effect=inputs), patch("axon.config.AxonConfig.save"):
        res = run_wizard()
        assert res["llm_model"] == "llama3.1:70b"
        assert res["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert res["vector_store"] == "qdrant"


def test_wizard_keyboard_interrupt():
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        res = run_wizard()
        assert res == {}
