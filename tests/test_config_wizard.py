from unittest.mock import patch

from axon.config_wizard import run_wizard


def test_wizard_abort():
    # Sequence: Mode="1", LLM Skip="n", Embedding Skip="n", Vector Store Skip="n", Save="n"
    with patch("builtins.input", side_effect=["1", "n", "n", "n", "n"]):
        res = run_wizard()
        assert res == {}


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
        "2",  # qdrant (DIFFERENT)
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
