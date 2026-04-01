"""Semantic and Table splitter tests for axon.splitters."""
from axon.splitters import CosineSemanticSplitter, SemanticTextSplitter, TableSplitter


def test_semantic_text_splitter_basic():
    splitter = SemanticTextSplitter(chunk_size=200, chunk_overlap=50)
    text = "This is a sentence. This is another sentence. And a third one to keep it interesting."
    chunks = splitter.split(text)
    assert len(chunks) >= 1


def test_table_splitter_csv():
    splitter = TableSplitter(table_name="TestTable")
    rows = [{"id": "1", "name": "test", "value": "10"}]
    headers = ["id", "name", "value"]
    chunks = splitter.transform_rows(rows, headers)
    assert len(chunks) >= 1
    assert "id: 1" in chunks[0]


def test_table_splitter_markdown():
    splitter = TableSplitter(table_name="TestTable")
    rows = [{"id": "1", "name": "test"}]
    headers = ["id", "name"]
    chunks = splitter.transform_rows(rows, headers)
    assert len(chunks) >= 1


def test_cosine_semantic_splitter_mock():
    # CosineSemanticSplitter requires an embed_fn
    from unittest.mock import MagicMock

    mock_embed = MagicMock()
    # It takes a list of strings and returns a list of embeddings
    mock_embed.embed.side_effect = lambda texts: [[0.1] * 384 for _ in texts]

    splitter = CosineSemanticSplitter(embed_fn=mock_embed.embed, max_chunk_size=200)
    text = "This is a sentence. This is another sentence. And a third one to keep it interesting."
    chunks = splitter.split(text)
    assert len(chunks) >= 1
