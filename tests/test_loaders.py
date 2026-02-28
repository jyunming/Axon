"""Tests for document loaders."""

import pytest
import tempfile
import json
from pathlib import Path
from rag_brain.loaders import TextLoader, JSONLoader, TSVLoader, DirectoryLoader


class TestTextLoader:
    """Test the TextLoader class."""

    def test_load_text_file(self):
        """Test loading a text file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document.")
            f.flush()

            loader = TextLoader()
            docs = loader.load(f.name)

            assert len(docs) == 1
            assert docs[0]["text"] == "This is a test document."
            assert docs[0]["metadata"]["type"] == "text"
            assert f.name in docs[0]["metadata"]["source"]


class TestJSONLoader:
    """Test the JSONLoader class."""

    def test_load_json_list(self):
        """Test loading JSON with a list of documents."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = [
                {"text": "Document 1", "author": "Alice"},
                {"text": "Document 2", "author": "Bob"}
            ]
            json.dump(data, f)
            f.flush()

            loader = JSONLoader()
            docs = loader.load(f.name)

            assert len(docs) == 2
            assert docs[0]["text"] == "Document 1"
            assert docs[0]["metadata"]["author"] == "Alice"

    def test_load_json_object(self):
        """Test loading JSON with a single object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {"text": "Single document", "category": "test"}
            json.dump(data, f)
            f.flush()

            loader = JSONLoader()
            docs = loader.load(f.name)

            assert len(docs) == 1
            assert docs[0]["text"] == "Single document"
            assert docs[0]["metadata"]["category"] == "test"


class TestDirectoryLoader:
    """Test the DirectoryLoader class."""

    def test_load_mixed_directory(self):
        """Test loading a directory with mixed file types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "test1.txt").write_text("Text file content")
            Path(tmpdir, "test2.json").write_text(json.dumps({"text": "JSON content"}))

            loader = DirectoryLoader()
            docs = loader.load(tmpdir)

            assert len(docs) >= 2
            texts = [doc["text"] for doc in docs]
            assert "Text file content" in texts
            assert "JSON content" in texts
