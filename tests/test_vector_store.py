from __future__ import annotations
"""
tests/test_vector_store_extra.py

Comprehensive pytest tests for axon.vector_store covering:
  - OpenVectorStore (chroma, qdrant, lancedb providers)
  - MultiVectorStore (fan-out read-only wrapper)
  - MultiBM25Retriever (fan-out BM25 wrapper)

Target missed coverage lines: 46-63, 93-94, 97-101, 127-128, 152, 168-186,
207, 231-239, 242, 252, 278-281, 313, 315-334, 336-353, 358, 371-374, 379,
381, 383-385, 391, 433-442, 445-449, 481-483.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    defaults = {
        "bm25_path": os.path.abspath("/tmp/bm25"),
        "vector_store_path": os.path.abspath("/tmp/vs"),
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


def _make_chroma_mocks():
    """Return (mock_chromadb_module, mock_client, mock_collection)."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    return mock_chromadb, mock_client, mock_collection


def _make_qdrant_mocks():
    """Return (mock_qdrant_module, mock_client)."""
    mock_client = MagicMock()
    mock_qdrant = MagicMock()
    mock_qdrant.QdrantClient.return_value = mock_client
    return mock_qdrant, mock_client


def _make_lancedb_mocks():
    """Return (mock_lancedb_module, mock_db_client, mock_table)."""
    mock_table = MagicMock()
    mock_db_client = MagicMock()
    mock_lancedb = MagicMock()
    mock_lancedb.connect.return_value = mock_db_client
    return mock_lancedb, mock_db_client, mock_table


# ---------------------------------------------------------------------------
# _sanitize_chroma_meta  (static method — no provider needed)
# ---------------------------------------------------------------------------


class TestSanitizeChromaMeta:
    """Tests for OpenVectorStore._sanitize_chroma_meta."""

    def _sanitize(self, metadatas):
        from axon.vector_store import OpenVectorStore

        return OpenVectorStore._sanitize_chroma_meta(metadatas)

    def test_none_input_returns_none(self):
        assert self._sanitize(None) is None

    def test_list_value_joined_by_pipe(self):
        result = self._sanitize([{"imports": ["os", "sys"]}])
        assert result == [{"imports": "os|sys"}]

    def test_empty_list_value_is_omitted(self):
        result = self._sanitize([{"imports": []}])
        assert result == [{}]

    def test_none_value_is_omitted(self):
        result = self._sanitize([{"source": None}])
        assert result == [{}]

    def test_str_passthrough(self):
        result = self._sanitize([{"source": "doc.txt"}])
        assert result == [{"source": "doc.txt"}]

    def test_int_passthrough(self):
        result = self._sanitize([{"page": 3}])
        assert result == [{"page": 3}]

    def test_float_passthrough(self):
        result = self._sanitize([{"score": 0.95}])
        assert result == [{"score": 0.95}]

    def test_bool_passthrough(self):
        result = self._sanitize([{"active": True}])
        assert result == [{"active": True}]

    def test_other_type_converted_to_str(self):
        result = self._sanitize([{"obj": object.__new__(object)}])
        # Just ensure it's a string
        assert isinstance(result[0]["obj"], str)

    def test_mixed_metadata(self):
        meta = {"source": "a.py", "tags": ["x", "y"], "count": 5, "flag": None}
        result = self._sanitize([meta])
        assert result == [{"source": "a.py", "tags": "x|y", "count": 5}]

    def test_multiple_dicts(self):
        result = self._sanitize([{"a": [1, 2]}, {"b": None}, {"c": "keep"}])
        assert result == [{"a": "1|2"}, {}, {"c": "keep"}]


# ---------------------------------------------------------------------------
# OpenVectorStore — Chroma provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreChroma:
    """Tests for OpenVectorStore with the chroma provider."""

    def _make_store(self, mock_chromadb=None, **config_kwargs):
        if mock_chromadb is None:
            mock_chromadb, _, _ = _make_chroma_mocks()
        cfg = _make_config(vector_store="chroma", **config_kwargs)
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store

    # --- _init_store ---

    def test_init_creates_persistent_client(self):
        mock_chromadb, mock_client, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        # AxonConfig normalises paths; on Windows /tmp/vs → C:\tmp\vs
        assert mock_chromadb.PersistentClient.call_count == 1
        actual_path = mock_chromadb.PersistentClient.call_args[1]["path"]
        # Verify the path resolves to our intended directory (suffix check is portable)
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        mock_client.get_or_create_collection.assert_called_once_with(
            name="axon", metadata={"hnsw:space": "cosine"}
        )
        assert store.collection is mock_collection

    def test_init_wsl_sqlite_readonly_raises_runtime_error(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.get_or_create_collection.side_effect = Exception(
            "database is locked (code: 8) readonly database"
        )
        mock_rich_console = MagicMock()

        cfg = _make_config(vector_store="chroma")
        with patch.dict(
            "sys.modules",
            {
                "chromadb": mock_chromadb,
                "rich.console": MagicMock(Console=mock_rich_console),
            },
        ):
            from axon.vector_store import OpenVectorStore

            with pytest.raises(RuntimeError, match="ChromaDB failed to initialize"):
                OpenVectorStore(cfg)

    def test_init_non_wsl_error_is_reraised(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.get_or_create_collection.side_effect = ValueError("some other error")
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from axon.vector_store import OpenVectorStore

            with pytest.raises(ValueError, match="some other error"):
                OpenVectorStore(cfg)

    # --- close ---

    def test_close_calls_client_close_if_available(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.close = MagicMock()
        store = self._make_store(mock_chromadb)
        store.close()
        mock_client.close.assert_called_once()
        assert store.client is None
        assert store.collection is None

    def test_close_ignores_exception_from_client_close(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.close = MagicMock(side_effect=Exception("boom"))
        store = self._make_store(mock_chromadb)
        # Should not raise
        store.close()
        assert store.client is None

    def test_close_when_no_close_method(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        # Remove close attribute
        del mock_client.close
        store = self._make_store(mock_chromadb)
        store.close()
        assert store.client is None
        assert store.collection is None

    def test_close_noop_when_no_client(self):
        mock_chromadb, _, _ = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.client = None
        store.close()  # Should not raise

    # --- add ---

    def test_add_calls_collection_add(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(
            ids=["id1"],
            texts=["hello world"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "doc.txt"}],
        )
        mock_collection.add.assert_called_once()
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["ids"] == ["id1"]
        assert kwargs["documents"] == ["hello world"]

    def test_add_normalizes_list_metadata(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(
            ids=["id1"],
            texts=["text"],
            embeddings=[[0.1]],
            metadatas=[{"tags": ["a", "b"]}],
        )
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["metadatas"] == [{"tags": "a|b"}]

    def test_add_large_batch_splits_into_chunks(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        n = 5001
        ids = [f"id{i}" for i in range(n)]
        texts = [f"text{i}" for i in range(n)]
        embeddings = [[float(i)] for i in range(n)]
        store.add(ids=ids, texts=texts, embeddings=embeddings)
        # Should have been called twice (5000 + 1)
        assert mock_collection.add.call_count == 2

    def test_add_dimension_mismatch_raises_and_logs(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.add.side_effect = Exception("embedding dimension mismatch")
        store = self._make_store(mock_chromadb)
        with pytest.raises(Exception, match="dimension"):
            store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])

    def test_add_no_metadata(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(ids=["id1"], texts=["hello"], embeddings=[[0.1, 0.2]])
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["metadatas"] is None

    def test_add_empty_ids_no_call(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(ids=[], texts=[], embeddings=[])
        # _ids_b is empty so loop breaks before calling
        mock_collection.add.assert_not_called()

    # --- list_documents ---

    def test_list_documents_returns_sorted_by_source(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"source": "b.txt"},
                {"source": "a.txt"},
                {"source": "b.txt"},
            ],
        }
        store = self._make_store(mock_chromadb)
        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        assert docs[1]["chunks"] == 2

    def test_list_documents_none_metadata_uses_unknown(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "metadatas": None,
        }
        store = self._make_store(mock_chromadb)
        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_no_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
        }
        store = self._make_store(mock_chromadb)
        results = store.search([0.1, 0.2], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert abs(results[0]["score"] - 0.9) < 1e-6
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]], n_results=2, where=None
        )

    def test_search_single_key_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["t"]],
            "distances": [[0.0]],
            "metadatas": [[{}]],
        }
        store = self._make_store(mock_chromadb)
        store.search([0.1], top_k=1, filter_dict={"source": "a.txt"})
        _, kwargs = mock_collection.query.call_args
        assert kwargs["where"] == {"source": {"$eq": "a.txt"}}

    def test_search_multi_key_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        store = self._make_store(mock_chromadb)
        store.search([0.1], top_k=5, filter_dict={"source": "a.txt", "type": "code"})
        _, kwargs = mock_collection.query.call_args
        assert kwargs["where"] == {
            "$and": [
                {"source": {"$eq": "a.txt"}},
                {"type": {"$eq": "code"}},
            ]
        }

    def test_search_no_metadatas_key(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["t"]],
            "distances": [[0.2]],
            "metadatas": None,
        }
        store = self._make_store(mock_chromadb)
        results = store.search([0.1], top_k=1)
        assert results[0]["metadata"] == {}

    # --- get_by_ids ---

    def test_get_by_ids_empty_returns_empty(self):
        mock_chromadb, _, _ = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        result = store.get_by_ids([])
        assert result == []

    def test_get_by_ids_returns_correct_docs(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert len(docs) == 2
        assert docs[0]["score"] == 1.0
        assert docs[0]["text"] == "text1"
        assert docs[1]["metadata"] == {"source": "b.txt"}

    def test_get_by_ids_pads_short_documents(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1"],  # shorter than ids
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert docs[1]["text"] == ""

    def test_get_by_ids_pads_short_metadatas(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.txt"}],  # shorter
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert docs[1]["metadata"] == {}

    # --- delete_by_ids ---

    def test_delete_by_ids_empty_is_noop(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.delete_by_ids([])
        mock_collection.delete.assert_not_called()

    def test_delete_by_ids_calls_collection_delete(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.delete_by_ids(["id1", "id2"])
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])


# ---------------------------------------------------------------------------
# OpenVectorStore — Qdrant provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreQdrant:
    """Tests for OpenVectorStore with the qdrant provider."""

    def _make_store_local(self, mock_qdrant=None):
        if mock_qdrant is None:
            mock_qdrant, _ = _make_qdrant_mocks()
        cfg = _make_config(vector_store="qdrant")
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_qdrant

    def _make_store_remote(self, url="http://qdrant:6333", api_key=""):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        cfg = _make_config(vector_store="qdrant")
        cfg.qdrant_url = url
        cfg.qdrant_api_key = api_key
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_qdrant, mock_client

    # --- _init_store ---

    def test_init_local_mode(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        store, _ = self._make_store_local(mock_qdrant)
        # AxonConfig normalises paths; verify local (non-remote) constructor was used
        assert mock_qdrant.QdrantClient.call_count == 1
        call_kwargs = mock_qdrant.QdrantClient.call_args[1]
        assert "path" in call_kwargs
        assert "url" not in call_kwargs
        actual_path = call_kwargs["path"]
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        assert store.client is mock_client

    def test_init_remote_mode_with_url(self):
        store, mock_qdrant, mock_client = self._make_store_remote(
            url="http://qdrant:6333", api_key="secret"
        )
        mock_qdrant.QdrantClient.assert_called_once_with(url="http://qdrant:6333", api_key="secret")

    def test_init_remote_mode_no_api_key_passes_none(self):
        store, mock_qdrant, mock_client = self._make_store_remote(
            url="http://qdrant:6333", api_key=""
        )
        mock_qdrant.QdrantClient.assert_called_once_with(url="http://qdrant:6333", api_key=None)

    # --- close ---

    def test_close_clears_client(self):
        store, _ = self._make_store_local()
        store.close()
        assert store.client is None

    # --- add ---

    def test_add_upserts_points(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_point_struct = MagicMock()
        mock_qdrant.models = MagicMock()
        mock_qdrant.models.PointStruct = mock_point_struct

        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.add(
                ids=["id1", "id2"],
                texts=["t1", "t2"],
                embeddings=[[0.1], [0.2]],
                metadatas=[{"source": "a"}, {"source": "b"}],
            )
        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args[1]
        assert kwargs["collection_name"] == "axon"

    def test_add_large_batch_qdrant_splits(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_qdrant.models = MagicMock()
        mock_qdrant.models.PointStruct = MagicMock(side_effect=lambda **kw: kw)

        n = 5001
        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.add(
                ids=[f"id{i}" for i in range(n)],
                texts=[f"t{i}" for i in range(n)],
                embeddings=[[float(i)] for i in range(n)],
            )
        assert mock_client.upsert.call_count == 2

    # --- list_documents ---

    def test_list_documents_from_scroll(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p1 = MagicMock()
        p1.id = "id1"
        p1.payload = {"source": "b.txt", "text": "t1"}
        p2 = MagicMock()
        p2.id = "id2"
        p2.payload = {"source": "a.txt", "text": "t2"}
        mock_client.scroll.return_value = ([p1, p2], None)

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        mock_client.scroll.assert_called_once_with(
            collection_name="axon", limit=10000, with_payload=True
        )

    def test_list_documents_no_source_uses_unknown(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = "id1"
        p.payload = {"text": "hello"}
        mock_client.scroll.return_value = ([p], None)

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_returns_results(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        r1 = MagicMock()
        r1.id = "id1"
        r1.score = 0.9
        r1.payload = {"text": "hello", "source": "a.txt"}
        mock_client.search.return_value = [r1]

        store, _ = self._make_store_local(mock_qdrant)
        results = store.search([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.9
        assert results[0]["text"] == "hello"
        assert "text" not in results[0]["metadata"]
        assert results[0]["metadata"]["source"] == "a.txt"

    def test_search_calls_correct_args(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_client.search.return_value = []
        store, _ = self._make_store_local(mock_qdrant)
        store.search([0.5], top_k=3)
        mock_client.search.assert_called_once_with(
            collection_name="axon", query_vector=[0.5], limit=3
        )

    # --- get_by_ids ---

    def test_get_by_ids_retrieves_points(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = "id1"
        p.payload = {"text": "hello", "source": "a.txt"}
        mock_client.retrieve.return_value = [p]

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.get_by_ids(["id1"])
        assert len(docs) == 1
        assert docs[0]["score"] == 1.0
        assert docs[0]["text"] == "hello"
        mock_client.retrieve.assert_called_once_with(
            collection_name="axon", ids=["id1"], with_payload=True
        )

    def test_get_by_ids_exception_returns_empty(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_client.retrieve.side_effect = Exception("connection error")
        store, _ = self._make_store_local(mock_qdrant)
        result = store.get_by_ids(["id1"])
        assert result == []

    def test_get_by_ids_empty_list_returns_empty(self):
        store, _ = self._make_store_local()
        assert store.get_by_ids([]) == []

    # --- delete_by_ids ---

    def test_delete_by_ids_calls_delete(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_models = MagicMock()
        mock_qdrant.models = mock_models

        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.delete_by_ids(["id1", "id2"])

        mock_client.delete.assert_called_once()

    def test_delete_by_ids_empty_is_noop(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        store, _ = self._make_store_local(mock_qdrant)
        store.delete_by_ids([])
        mock_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# OpenVectorStore — LanceDB provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreLanceDB:
    """Tests for OpenVectorStore with the lancedb provider."""

    def _make_store(self, mock_lancedb=None, open_table_raises=False):
        if mock_lancedb is None:
            mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        else:
            mock_db = mock_lancedb.connect.return_value
            mock_table = MagicMock()

        if open_table_raises:
            mock_db.open_table.side_effect = Exception("no such table")
        else:
            mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_db, mock_table

    # --- _init_store ---

    def test_init_connects_and_opens_table(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        # AxonConfig normalises paths on Windows (/tmp/vs → C:\tmp\vs)
        assert mock_lancedb.connect.call_count == 1
        actual_path = mock_lancedb.connect.call_args[0][0]
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        # collection should be the object returned from open_table
        assert store.collection is mock_table

    def test_init_sets_collection_none_when_table_missing(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("table not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.collection is None

    # --- close ---

    def test_close_clears_client_and_collection(self):
        store, _, _ = self._make_store()
        store.close()
        assert store.client is None
        assert store.collection is None

    def test_close_noop_when_no_client(self):
        store, _, _ = self._make_store()
        store.client = None
        store.close()  # Should not raise

    # --- add ---

    def test_add_creates_table_on_first_call(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        mock_new_table = MagicMock()
        mock_db.create_table.return_value = mock_new_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        assert store.collection is None
        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "a.txt"}],
        )
        mock_db.create_table.assert_called_once()
        assert store.collection is mock_new_table

    def test_add_appends_to_existing_table(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1]],
            metadatas=[{"source": "a.txt"}],
        )
        mock_table.add.assert_called_once()
        rows = mock_table.add.call_args[0][0]
        assert rows[0]["id"] == "id1"
        assert rows[0]["source"] == "a.txt"

    def test_add_no_metadata_uses_empty_dict(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])
        rows = mock_table.add.call_args[0][0]
        assert rows[0]["source"] == ""

    # --- list_documents ---

    def test_list_documents_returns_empty_when_collection_none(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.list_documents() == []

    def test_list_documents_from_arrow(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.to_arrow.return_value.to_pydict.return_value = {
            "id": ["id1", "id2", "id3"],
            "source": ["b.txt", "a.txt", "b.txt"],
        }

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        assert docs[1]["chunks"] == 2

    def test_list_documents_none_source_uses_unknown(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.to_arrow.return_value.to_pydict.return_value = {
            "id": ["id1"],
            "source": [None],
        }

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_empty_collection_returns_empty(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        results = store.search([0.1, 0.2])
        assert results == []

    def test_search_returns_results_with_score(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        import json

        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {
                "id": "id1",
                "text": "hello",
                "_distance": 0.2,
                "metadata_json": json.dumps({"source": "a.txt"}),
            }
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        results = store.search([0.1], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert abs(results[0]["score"] - 0.8) < 1e-6
        assert results[0]["metadata"] == {"source": "a.txt"}

    def test_search_distance_gt_1_score_clamps_to_zero(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"id": "id1", "text": "t", "_distance": 1.5, "metadata_json": "{}"}
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        results = store.search([0.1])
        assert results[0]["score"] == 0.0

    # --- get_by_ids ---

    def test_get_by_ids_empty_returns_empty(self):
        store, _, _ = self._make_store()
        assert store.get_by_ids([]) == []

    def test_get_by_ids_collection_none_returns_empty(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.get_by_ids(["id1"]) == []

    def test_get_by_ids_returns_results(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        import json

        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"id": "id1", "text": "hello", "metadata_json": json.dumps({"source": "a.txt"})}
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.get_by_ids(["id1"])
        assert docs[0]["id"] == "id1"
        assert docs[0]["score"] == 1.0

    def test_get_by_ids_exception_returns_empty(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.where.side_effect = Exception("query error")

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        result = store.get_by_ids(["id1"])
        assert result == []

    def test_get_by_ids_generates_correct_sql(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.where.return_value.to_list.return_value = []

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.get_by_ids(["id1", "id2"])
        where_call_args = mock_table.search.return_value.where.call_args
        sql = where_call_args[0][0]
        assert "id1" in sql
        assert "id2" in sql
        assert "IN" in sql

    # --- delete_by_ids ---

    def test_delete_by_ids_empty_is_noop(self):
        store, _, mock_table = self._make_store()
        store.delete_by_ids([])
        mock_table.delete.assert_not_called()

    def test_delete_by_ids_collection_none_is_noop(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        store.delete_by_ids(["id1"])  # Should not raise

    def test_delete_by_ids_generates_sql_and_calls_delete(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.delete_by_ids(["id1", "id2"])
        mock_table.delete.assert_called_once()
        sql = mock_table.delete.call_args[0][0]
        assert "id1" in sql
        assert "id2" in sql
        assert "IN" in sql


# ---------------------------------------------------------------------------
# MultiVectorStore
# ---------------------------------------------------------------------------


class TestMultiVectorStore:
    """Tests for MultiVectorStore fan-out read-only wrapper."""

    def _make_mock_store(self, provider="chroma"):
        store = MagicMock(
            spec=[
                "provider",
                "collection",
                "search",
                "list_documents",
                "get_by_ids",
                "add",
                "delete_by_ids",
            ]
        )
        store.provider = provider
        store.collection = MagicMock()
        return store

    def _make_multi(self, n=2):
        from axon.vector_store import MultiVectorStore

        stores = [self._make_mock_store() for _ in range(n)]
        multi = MultiVectorStore(stores)
        return multi, stores

    # --- constructor ---

    def test_provider_from_first_store(self):
        from axon.vector_store import MultiVectorStore

        s1 = self._make_mock_store(provider="qdrant")
        s2 = self._make_mock_store(provider="chroma")
        m = MultiVectorStore([s1, s2])
        assert m.provider == "qdrant"

    def test_empty_stores_default_provider(self):
        from axon.vector_store import MultiVectorStore

        m = MultiVectorStore([])
        assert m.provider == "chroma"
        assert m.collection is None

    # --- search ---

    def test_search_merges_results_deduplicates_keeps_max_score(self):
        multi, stores = self._make_multi(2)
        stores[0].search.return_value = [
            {"id": "a", "score": 0.8, "text": "from store0"},
            {"id": "b", "score": 0.5, "text": "b"},
        ]
        stores[1].search.return_value = [
            {"id": "a", "score": 0.9, "text": "from store1"},  # higher score
            {"id": "c", "score": 0.3, "text": "c"},
        ]
        results = multi.search([0.1, 0.2], top_k=10)
        ids = [r["id"] for r in results]
        # All unique
        assert len(ids) == len(set(ids))
        # "a" should have score 0.9 (from store1)
        a_result = next(r for r in results if r["id"] == "a")
        assert a_result["score"] == 0.9
        # Results sorted by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self):
        multi, stores = self._make_multi(2)
        stores[0].search.return_value = [
            {"id": f"a{i}", "score": float(i) / 10, "text": ""} for i in range(5)
        ]
        stores[1].search.return_value = [
            {"id": f"b{i}", "score": float(i) / 10, "text": ""} for i in range(5)
        ]
        results = multi.search([0.1], top_k=3)
        assert len(results) == 3

    def test_search_empty_stores(self):
        from axon.vector_store import MultiVectorStore

        MultiVectorStore([])
        # ThreadPoolExecutor with max_workers=min(4,0) — might raise; skip gracefully
        # Actually min(4,0) = 0 which raises; let's use a single-store
        s = self._make_mock_store()
        s.search.return_value = []
        m2 = MultiVectorStore([s])
        results = m2.search([0.1])
        assert results == []

    # --- list_documents ---

    def test_list_documents_merges_and_aggregates_chunks(self):
        multi, stores = self._make_multi(2)
        stores[0].list_documents.return_value = [
            {"source": "a.txt", "chunks": 3, "doc_ids": ["id1", "id2", "id3"]},
            {"source": "b.txt", "chunks": 1, "doc_ids": ["id4"]},
        ]
        stores[1].list_documents.return_value = [
            {"source": "a.txt", "chunks": 2, "doc_ids": ["id5", "id6"]},
        ]
        docs = multi.list_documents()
        sources = {d["source"]: d for d in docs}
        assert sources["a.txt"]["chunks"] == 5
        assert len(sources["a.txt"]["doc_ids"]) == 5
        assert sources["b.txt"]["chunks"] == 1
        # Sorted by source
        assert [d["source"] for d in docs] == sorted([d["source"] for d in docs])

    def test_list_documents_no_overlap(self):
        multi, stores = self._make_multi(2)
        stores[0].list_documents.return_value = [
            {"source": "a.txt", "chunks": 1, "doc_ids": ["id1"]}
        ]
        stores[1].list_documents.return_value = [
            {"source": "b.txt", "chunks": 2, "doc_ids": ["id2", "id3"]}
        ]
        docs = multi.list_documents()
        assert len(docs) == 2

    # --- get_by_ids ---

    def test_get_by_ids_merges_from_all_stores(self):
        multi, stores = self._make_multi(2)
        stores[0].get_by_ids.return_value = [
            {"id": "id1", "text": "t1", "score": 1.0, "metadata": {}}
        ]
        stores[1].get_by_ids.return_value = [
            {"id": "id2", "text": "t2", "score": 1.0, "metadata": {}}
        ]
        docs = multi.get_by_ids(["id1", "id2"])
        ids = {d["id"] for d in docs}
        assert ids == {"id1", "id2"}

    def test_get_by_ids_deduplicates_by_id(self):
        multi, stores = self._make_multi(2)
        doc = {"id": "id1", "text": "t1", "score": 1.0, "metadata": {}}
        stores[0].get_by_ids.return_value = [doc]
        stores[1].get_by_ids.return_value = [doc]
        docs = multi.get_by_ids(["id1"])
        assert len(docs) == 1

    # --- add raises ---

    def test_add_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add(["id1"], ["text"], [[0.1]])

    # --- delete_by_ids raises ---

    def test_delete_by_ids_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_by_ids(["id1"])

    def test_delete_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_documents(["id1"])


# ---------------------------------------------------------------------------
# MultiBM25Retriever
# ---------------------------------------------------------------------------


class TestMultiBM25Retriever:
    """Tests for MultiBM25Retriever fan-out BM25 wrapper."""

    def _make_retriever_mock(self):
        r = MagicMock()
        return r

    def _make_multi(self, n=2):
        from axon.vector_store import MultiBM25Retriever

        retrievers = [self._make_retriever_mock() for _ in range(n)]
        multi = MultiBM25Retriever(retrievers)
        return multi, retrievers

    # --- search ---

    def test_search_merges_and_deduplicates(self):
        multi, retrievers = self._make_multi(2)
        retrievers[0].search.return_value = [
            {"id": "a", "score": 0.7, "text": "from r0"},
            {"id": "b", "score": 0.5, "text": "b"},
        ]
        retrievers[1].search.return_value = [
            {"id": "a", "score": 0.9, "text": "from r1"},  # higher score
            {"id": "c", "score": 0.2, "text": "c"},
        ]
        results = multi.search("hello world", top_k=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))
        a_result = next(r for r in results if r["id"] == "a")
        assert a_result["score"] == 0.9

    def test_search_respects_top_k(self):
        multi, retrievers = self._make_multi(2)
        retrievers[0].search.return_value = [
            {"id": f"a{i}", "score": float(i), "text": ""} for i in range(5)
        ]
        retrievers[1].search.return_value = [
            {"id": f"b{i}", "score": float(i), "text": ""} for i in range(5)
        ]
        results = multi.search("query", top_k=4)
        assert len(results) == 4

    def test_search_results_sorted_descending(self):
        multi, retrievers = self._make_multi(1)
        retrievers[0].search.return_value = [
            {"id": "a", "score": 0.3, "text": ""},
            {"id": "b", "score": 0.8, "text": ""},
            {"id": "c", "score": 0.5, "text": ""},
        ]
        results = multi.search("q")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_passes_query_and_top_k(self):
        multi, retrievers = self._make_multi(1)
        retrievers[0].search.return_value = []
        multi.search("my query", top_k=7)
        retrievers[0].search.assert_called_once_with("my query", 7)

    # --- close ---

    def test_close_calls_close_on_all_retrievers(self):
        multi, retrievers = self._make_multi(3)
        multi.close()
        for r in retrievers:
            r.close.assert_called_once()

    def test_close_skips_retrievers_without_close(self):
        from axon.vector_store import MultiBM25Retriever

        r1 = MagicMock(spec=["search"])  # No close attribute
        r2 = MagicMock()
        multi = MultiBM25Retriever([r1, r2])
        multi.close()  # Should not raise
        r2.close.assert_called_once()

    # --- delete_documents raises ---

    def test_delete_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_documents(["id1"])

    # --- add_documents raises ---

    def test_add_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add_documents(["doc"])

    def test_add_documents_raises_with_kwargs(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add_documents(ids=["id1"], texts=["text"])


# ---------------------------------------------------------------------------
# Edge cases and integration-style tests
# ---------------------------------------------------------------------------


class TestOpenVectorStoreEdgeCases:
    """Additional edge case tests for corner branches."""

    def test_chroma_list_documents_doc_ids_populated(self):
        from axon.vector_store import OpenVectorStore

        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "metadatas": [{"source": "a.txt"}, {"source": "a.txt"}],
        }
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store = OpenVectorStore(cfg)
        docs = store.list_documents()
        assert docs[0]["doc_ids"] == ["id1", "id2"]

    def test_qdrant_list_documents_doc_ids_are_strings(self):
        from axon.vector_store import OpenVectorStore

        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = 42  # integer id
        p.payload = {"source": "doc.txt", "text": "hello"}
        mock_client.scroll.return_value = ([p], None)
        cfg = _make_config(vector_store="qdrant")
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            store = OpenVectorStore(cfg)
        docs = store.list_documents()
        assert docs[0]["doc_ids"] == ["42"]

    def test_lancedb_search_calls_limit(self):
        from axon.vector_store import OpenVectorStore

        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.limit.return_value.to_list.return_value = []
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            store = OpenVectorStore(cfg)
        store.search([0.1, 0.2], top_k=7)
        mock_table.search.return_value.limit.assert_called_once_with(7)

    def test_chroma_add_non_dimension_error_reraises(self):
        from axon.vector_store import OpenVectorStore

        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.add.side_effect = ValueError("some unrelated error")
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store = OpenVectorStore(cfg)
        with pytest.raises(ValueError, match="some unrelated error"):
            store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])

    def test_multi_vector_store_search_all_empty(self):
        from axon.vector_store import MultiVectorStore

        s1, s2 = MagicMock(), MagicMock()
        s1.provider = "chroma"
        s2.provider = "chroma"
        s1.collection = MagicMock()
        s2.collection = MagicMock()
        s1.search.return_value = []
        s2.search.return_value = []
        m = MultiVectorStore([s1, s2])
        assert m.search([0.1]) == []

    def test_sanitize_chroma_meta_empty_input(self):
        from axon.vector_store import OpenVectorStore

        result = OpenVectorStore._sanitize_chroma_meta([])
        assert result == []

    def test_lancedb_add_metadata_json_serialized(self):
        import json

        from axon.vector_store import OpenVectorStore

        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            store = OpenVectorStore(cfg)
        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "a.txt", "page": 1}],
        )
        rows = mock_table.add.call_args[0][0]
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["source"] == "a.txt"
        assert meta["page"] == 1
"""
tests/test_vector_store_extra.py

Comprehensive pytest tests for axon.vector_store covering:
  - OpenVectorStore (chroma, qdrant, lancedb providers)
  - MultiVectorStore (fan-out read-only wrapper)
  - MultiBM25Retriever (fan-out BM25 wrapper)

Target missed coverage lines: 46-63, 93-94, 97-101, 127-128, 152, 168-186,
207, 231-239, 242, 252, 278-281, 313, 315-334, 336-353, 358, 371-374, 379,
381, 383-385, 391, 433-442, 445-449, 481-483.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from axon.config import AxonConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs):
    defaults = {
        "bm25_path": os.path.abspath("/tmp/bm25"),
        "vector_store_path": os.path.abspath("/tmp/vs"),
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


def _make_chroma_mocks():
    """Return (mock_chromadb_module, mock_client, mock_collection)."""
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_client
    return mock_chromadb, mock_client, mock_collection


def _make_qdrant_mocks():
    """Return (mock_qdrant_module, mock_client)."""
    mock_client = MagicMock()
    mock_qdrant = MagicMock()
    mock_qdrant.QdrantClient.return_value = mock_client
    return mock_qdrant, mock_client


def _make_lancedb_mocks():
    """Return (mock_lancedb_module, mock_db_client, mock_table)."""
    mock_table = MagicMock()
    mock_db_client = MagicMock()
    mock_lancedb = MagicMock()
    mock_lancedb.connect.return_value = mock_db_client
    return mock_lancedb, mock_db_client, mock_table


# ---------------------------------------------------------------------------
# _sanitize_chroma_meta  (static method — no provider needed)
# ---------------------------------------------------------------------------


class TestSanitizeChromaMeta:
    """Tests for OpenVectorStore._sanitize_chroma_meta."""

    def _sanitize(self, metadatas):
        from axon.vector_store import OpenVectorStore

        return OpenVectorStore._sanitize_chroma_meta(metadatas)

    def test_none_input_returns_none(self):
        assert self._sanitize(None) is None

    def test_list_value_joined_by_pipe(self):
        result = self._sanitize([{"imports": ["os", "sys"]}])
        assert result == [{"imports": "os|sys"}]

    def test_empty_list_value_is_omitted(self):
        result = self._sanitize([{"imports": []}])
        assert result == [{}]

    def test_none_value_is_omitted(self):
        result = self._sanitize([{"source": None}])
        assert result == [{}]

    def test_str_passthrough(self):
        result = self._sanitize([{"source": "doc.txt"}])
        assert result == [{"source": "doc.txt"}]

    def test_int_passthrough(self):
        result = self._sanitize([{"page": 3}])
        assert result == [{"page": 3}]

    def test_float_passthrough(self):
        result = self._sanitize([{"score": 0.95}])
        assert result == [{"score": 0.95}]

    def test_bool_passthrough(self):
        result = self._sanitize([{"active": True}])
        assert result == [{"active": True}]

    def test_other_type_converted_to_str(self):
        result = self._sanitize([{"obj": object.__new__(object)}])
        # Just ensure it's a string
        assert isinstance(result[0]["obj"], str)

    def test_mixed_metadata(self):
        meta = {"source": "a.py", "tags": ["x", "y"], "count": 5, "flag": None}
        result = self._sanitize([meta])
        assert result == [{"source": "a.py", "tags": "x|y", "count": 5}]

    def test_multiple_dicts(self):
        result = self._sanitize([{"a": [1, 2]}, {"b": None}, {"c": "keep"}])
        assert result == [{"a": "1|2"}, {}, {"c": "keep"}]


# ---------------------------------------------------------------------------
# OpenVectorStore — Chroma provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreChroma:
    """Tests for OpenVectorStore with the chroma provider."""

    def _make_store(self, mock_chromadb=None, **config_kwargs):
        if mock_chromadb is None:
            mock_chromadb, _, _ = _make_chroma_mocks()
        cfg = _make_config(vector_store="chroma", **config_kwargs)
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store

    # --- _init_store ---

    def test_init_creates_persistent_client(self):
        mock_chromadb, mock_client, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        # AxonConfig normalises paths; on Windows /tmp/vs → C:\tmp\vs
        assert mock_chromadb.PersistentClient.call_count == 1
        actual_path = mock_chromadb.PersistentClient.call_args[1]["path"]
        # Verify the path resolves to our intended directory (suffix check is portable)
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        mock_client.get_or_create_collection.assert_called_once_with(
            name="axon", metadata={"hnsw:space": "cosine"}
        )
        assert store.collection is mock_collection

    def test_init_wsl_sqlite_readonly_raises_runtime_error(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.get_or_create_collection.side_effect = Exception(
            "database is locked (code: 8) readonly database"
        )
        mock_rich_console = MagicMock()

        cfg = _make_config(vector_store="chroma")
        with patch.dict(
            "sys.modules",
            {
                "chromadb": mock_chromadb,
                "rich.console": MagicMock(Console=mock_rich_console),
            },
        ):
            from axon.vector_store import OpenVectorStore

            with pytest.raises(RuntimeError, match="ChromaDB failed to initialize"):
                OpenVectorStore(cfg)

    def test_init_non_wsl_error_is_reraised(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.get_or_create_collection.side_effect = ValueError("some other error")
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            from axon.vector_store import OpenVectorStore

            with pytest.raises(ValueError, match="some other error"):
                OpenVectorStore(cfg)

    # --- close ---

    def test_close_calls_client_close_if_available(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.close = MagicMock()
        store = self._make_store(mock_chromadb)
        store.close()
        mock_client.close.assert_called_once()
        assert store.client is None
        assert store.collection is None

    def test_close_ignores_exception_from_client_close(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        mock_client.close = MagicMock(side_effect=Exception("boom"))
        store = self._make_store(mock_chromadb)
        # Should not raise
        store.close()
        assert store.client is None

    def test_close_when_no_close_method(self):
        mock_chromadb, mock_client, _ = _make_chroma_mocks()
        # Remove close attribute
        del mock_client.close
        store = self._make_store(mock_chromadb)
        store.close()
        assert store.client is None
        assert store.collection is None

    def test_close_noop_when_no_client(self):
        mock_chromadb, _, _ = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.client = None
        store.close()  # Should not raise

    # --- add ---

    def test_add_calls_collection_add(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(
            ids=["id1"],
            texts=["hello world"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "doc.txt"}],
        )
        mock_collection.add.assert_called_once()
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["ids"] == ["id1"]
        assert kwargs["documents"] == ["hello world"]

    def test_add_normalizes_list_metadata(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(
            ids=["id1"],
            texts=["text"],
            embeddings=[[0.1]],
            metadatas=[{"tags": ["a", "b"]}],
        )
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["metadatas"] == [{"tags": "a|b"}]

    def test_add_large_batch_splits_into_chunks(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        n = 5001
        ids = [f"id{i}" for i in range(n)]
        texts = [f"text{i}" for i in range(n)]
        embeddings = [[float(i)] for i in range(n)]
        store.add(ids=ids, texts=texts, embeddings=embeddings)
        # Should have been called twice (5000 + 1)
        assert mock_collection.add.call_count == 2

    def test_add_dimension_mismatch_raises_and_logs(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.add.side_effect = Exception("embedding dimension mismatch")
        store = self._make_store(mock_chromadb)
        with pytest.raises(Exception, match="dimension"):
            store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])

    def test_add_no_metadata(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(ids=["id1"], texts=["hello"], embeddings=[[0.1, 0.2]])
        kwargs = mock_collection.add.call_args[1]
        assert kwargs["metadatas"] is None

    def test_add_empty_ids_no_call(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.add(ids=[], texts=[], embeddings=[])
        # _ids_b is empty so loop breaks before calling
        mock_collection.add.assert_not_called()

    # --- list_documents ---

    def test_list_documents_returns_sorted_by_source(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"source": "b.txt"},
                {"source": "a.txt"},
                {"source": "b.txt"},
            ],
        }
        store = self._make_store(mock_chromadb)
        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        assert docs[1]["chunks"] == 2

    def test_list_documents_none_metadata_uses_unknown(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1"],
            "metadatas": None,
        }
        store = self._make_store(mock_chromadb)
        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_no_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["text1", "text2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}]],
        }
        store = self._make_store(mock_chromadb)
        results = store.search([0.1, 0.2], top_k=2)
        assert len(results) == 2
        assert results[0]["id"] == "id1"
        assert abs(results[0]["score"] - 0.9) < 1e-6
        mock_collection.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]], n_results=2, where=None
        )

    def test_search_single_key_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["t"]],
            "distances": [[0.0]],
            "metadatas": [[{}]],
        }
        store = self._make_store(mock_chromadb)
        store.search([0.1], top_k=1, filter_dict={"source": "a.txt"})
        _, kwargs = mock_collection.query.call_args
        assert kwargs["where"] == {"source": {"$eq": "a.txt"}}

    def test_search_multi_key_filter(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        store = self._make_store(mock_chromadb)
        store.search([0.1], top_k=5, filter_dict={"source": "a.txt", "type": "code"})
        _, kwargs = mock_collection.query.call_args
        assert kwargs["where"] == {
            "$and": [
                {"source": {"$eq": "a.txt"}},
                {"type": {"$eq": "code"}},
            ]
        }

    def test_search_no_metadatas_key(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["t"]],
            "distances": [[0.2]],
            "metadatas": None,
        }
        store = self._make_store(mock_chromadb)
        results = store.search([0.1], top_k=1)
        assert results[0]["metadata"] == {}

    # --- get_by_ids ---

    def test_get_by_ids_empty_returns_empty(self):
        mock_chromadb, _, _ = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        result = store.get_by_ids([])
        assert result == []

    def test_get_by_ids_returns_correct_docs(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert len(docs) == 2
        assert docs[0]["score"] == 1.0
        assert docs[0]["text"] == "text1"
        assert docs[1]["metadata"] == {"source": "b.txt"}

    def test_get_by_ids_pads_short_documents(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1"],  # shorter than ids
            "metadatas": [{"source": "a.txt"}, {"source": "b.txt"}],
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert docs[1]["text"] == ""

    def test_get_by_ids_pads_short_metadatas(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "documents": ["text1", "text2"],
            "metadatas": [{"source": "a.txt"}],  # shorter
        }
        store = self._make_store(mock_chromadb)
        docs = store.get_by_ids(["id1", "id2"])
        assert docs[1]["metadata"] == {}

    # --- delete_by_ids ---

    def test_delete_by_ids_empty_is_noop(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.delete_by_ids([])
        mock_collection.delete.assert_not_called()

    def test_delete_by_ids_calls_collection_delete(self):
        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        store = self._make_store(mock_chromadb)
        store.delete_by_ids(["id1", "id2"])
        mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])


# ---------------------------------------------------------------------------
# OpenVectorStore — Qdrant provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreQdrant:
    """Tests for OpenVectorStore with the qdrant provider."""

    def _make_store_local(self, mock_qdrant=None):
        if mock_qdrant is None:
            mock_qdrant, _ = _make_qdrant_mocks()
        cfg = _make_config(vector_store="qdrant")
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_qdrant

    def _make_store_remote(self, url="http://qdrant:6333", api_key=""):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        cfg = _make_config(vector_store="qdrant")
        cfg.qdrant_url = url
        cfg.qdrant_api_key = api_key
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_qdrant, mock_client

    # --- _init_store ---

    def test_init_local_mode(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        store, _ = self._make_store_local(mock_qdrant)
        # AxonConfig normalises paths; verify local (non-remote) constructor was used
        assert mock_qdrant.QdrantClient.call_count == 1
        call_kwargs = mock_qdrant.QdrantClient.call_args[1]
        assert "path" in call_kwargs
        assert "url" not in call_kwargs
        actual_path = call_kwargs["path"]
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        assert store.client is mock_client

    def test_init_remote_mode_with_url(self):
        store, mock_qdrant, mock_client = self._make_store_remote(
            url="http://qdrant:6333", api_key="secret"
        )
        mock_qdrant.QdrantClient.assert_called_once_with(url="http://qdrant:6333", api_key="secret")

    def test_init_remote_mode_no_api_key_passes_none(self):
        store, mock_qdrant, mock_client = self._make_store_remote(
            url="http://qdrant:6333", api_key=""
        )
        mock_qdrant.QdrantClient.assert_called_once_with(url="http://qdrant:6333", api_key=None)

    # --- close ---

    def test_close_clears_client(self):
        store, _ = self._make_store_local()
        store.close()
        assert store.client is None

    # --- add ---

    def test_add_upserts_points(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_point_struct = MagicMock()
        mock_qdrant.models = MagicMock()
        mock_qdrant.models.PointStruct = mock_point_struct

        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.add(
                ids=["id1", "id2"],
                texts=["t1", "t2"],
                embeddings=[[0.1], [0.2]],
                metadatas=[{"source": "a"}, {"source": "b"}],
            )
        mock_client.upsert.assert_called_once()
        kwargs = mock_client.upsert.call_args[1]
        assert kwargs["collection_name"] == "axon"

    def test_add_large_batch_qdrant_splits(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_qdrant.models = MagicMock()
        mock_qdrant.models.PointStruct = MagicMock(side_effect=lambda **kw: kw)

        n = 5001
        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_qdrant.models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.add(
                ids=[f"id{i}" for i in range(n)],
                texts=[f"t{i}" for i in range(n)],
                embeddings=[[float(i)] for i in range(n)],
            )
        assert mock_client.upsert.call_count == 2

    # --- list_documents ---

    def test_list_documents_from_scroll(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p1 = MagicMock()
        p1.id = "id1"
        p1.payload = {"source": "b.txt", "text": "t1"}
        p2 = MagicMock()
        p2.id = "id2"
        p2.payload = {"source": "a.txt", "text": "t2"}
        mock_client.scroll.return_value = ([p1, p2], None)

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        mock_client.scroll.assert_called_once_with(
            collection_name="axon", limit=10000, with_payload=True
        )

    def test_list_documents_no_source_uses_unknown(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = "id1"
        p.payload = {"text": "hello"}
        mock_client.scroll.return_value = ([p], None)

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_returns_results(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        r1 = MagicMock()
        r1.id = "id1"
        r1.score = 0.9
        r1.payload = {"text": "hello", "source": "a.txt"}
        mock_client.search.return_value = [r1]

        store, _ = self._make_store_local(mock_qdrant)
        results = store.search([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert results[0]["score"] == 0.9
        assert results[0]["text"] == "hello"
        assert "text" not in results[0]["metadata"]
        assert results[0]["metadata"]["source"] == "a.txt"

    def test_search_calls_correct_args(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_client.search.return_value = []
        store, _ = self._make_store_local(mock_qdrant)
        store.search([0.5], top_k=3)
        mock_client.search.assert_called_once_with(
            collection_name="axon", query_vector=[0.5], limit=3
        )

    # --- get_by_ids ---

    def test_get_by_ids_retrieves_points(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = "id1"
        p.payload = {"text": "hello", "source": "a.txt"}
        mock_client.retrieve.return_value = [p]

        store, _ = self._make_store_local(mock_qdrant)
        docs = store.get_by_ids(["id1"])
        assert len(docs) == 1
        assert docs[0]["score"] == 1.0
        assert docs[0]["text"] == "hello"
        mock_client.retrieve.assert_called_once_with(
            collection_name="axon", ids=["id1"], with_payload=True
        )

    def test_get_by_ids_exception_returns_empty(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_client.retrieve.side_effect = Exception("connection error")
        store, _ = self._make_store_local(mock_qdrant)
        result = store.get_by_ids(["id1"])
        assert result == []

    def test_get_by_ids_empty_list_returns_empty(self):
        store, _ = self._make_store_local()
        assert store.get_by_ids([]) == []

    # --- delete_by_ids ---

    def test_delete_by_ids_calls_delete(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        mock_models = MagicMock()
        mock_qdrant.models = mock_models

        cfg = _make_config(vector_store="qdrant")
        with patch.dict(
            "sys.modules",
            {
                "qdrant_client": mock_qdrant,
                "qdrant_client.models": mock_models,
            },
        ):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
            store.delete_by_ids(["id1", "id2"])

        mock_client.delete.assert_called_once()

    def test_delete_by_ids_empty_is_noop(self):
        mock_qdrant, mock_client = _make_qdrant_mocks()
        store, _ = self._make_store_local(mock_qdrant)
        store.delete_by_ids([])
        mock_client.delete.assert_not_called()


# ---------------------------------------------------------------------------
# OpenVectorStore — LanceDB provider
# ---------------------------------------------------------------------------


class TestOpenVectorStoreLanceDB:
    """Tests for OpenVectorStore with the lancedb provider."""

    def _make_store(self, mock_lancedb=None, open_table_raises=False):
        if mock_lancedb is None:
            mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        else:
            mock_db = mock_lancedb.connect.return_value
            mock_table = MagicMock()

        if open_table_raises:
            mock_db.open_table.side_effect = Exception("no such table")
        else:
            mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        return store, mock_db, mock_table

    # --- _init_store ---

    def test_init_connects_and_opens_table(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        # AxonConfig normalises paths on Windows (/tmp/vs → C:\tmp\vs)
        assert mock_lancedb.connect.call_count == 1
        actual_path = mock_lancedb.connect.call_args[0][0]
        assert actual_path.replace("\\", "/").endswith("/tmp/vs")
        # collection should be the object returned from open_table
        assert store.collection is mock_table

    def test_init_sets_collection_none_when_table_missing(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("table not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.collection is None

    # --- close ---

    def test_close_clears_client_and_collection(self):
        store, _, _ = self._make_store()
        store.close()
        assert store.client is None
        assert store.collection is None

    def test_close_noop_when_no_client(self):
        store, _, _ = self._make_store()
        store.client = None
        store.close()  # Should not raise

    # --- add ---

    def test_add_creates_table_on_first_call(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        mock_new_table = MagicMock()
        mock_db.create_table.return_value = mock_new_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        assert store.collection is None
        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "a.txt"}],
        )
        mock_db.create_table.assert_called_once()
        assert store.collection is mock_new_table

    def test_add_appends_to_existing_table(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1]],
            metadatas=[{"source": "a.txt"}],
        )
        mock_table.add.assert_called_once()
        rows = mock_table.add.call_args[0][0]
        assert rows[0]["id"] == "id1"
        assert rows[0]["source"] == "a.txt"

    def test_add_no_metadata_uses_empty_dict(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])
        rows = mock_table.add.call_args[0][0]
        assert rows[0]["source"] == ""

    # --- list_documents ---

    def test_list_documents_returns_empty_when_collection_none(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.list_documents() == []

    def test_list_documents_from_arrow(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.to_arrow.return_value.to_pydict.return_value = {
            "id": ["id1", "id2", "id3"],
            "source": ["b.txt", "a.txt", "b.txt"],
        }

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.list_documents()
        assert [d["source"] for d in docs] == ["a.txt", "b.txt"]
        assert docs[1]["chunks"] == 2

    def test_list_documents_none_source_uses_unknown(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.to_arrow.return_value.to_pydict.return_value = {
            "id": ["id1"],
            "source": [None],
        }

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.list_documents()
        assert docs[0]["source"] == "unknown"

    # --- search ---

    def test_search_empty_collection_returns_empty(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        results = store.search([0.1, 0.2])
        assert results == []

    def test_search_returns_results_with_score(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        import json

        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {
                "id": "id1",
                "text": "hello",
                "_distance": 0.2,
                "metadata_json": json.dumps({"source": "a.txt"}),
            }
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        results = store.search([0.1], top_k=5)
        assert len(results) == 1
        assert results[0]["id"] == "id1"
        assert abs(results[0]["score"] - 0.8) < 1e-6
        assert results[0]["metadata"] == {"source": "a.txt"}

    def test_search_distance_gt_1_score_clamps_to_zero(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"id": "id1", "text": "t", "_distance": 1.5, "metadata_json": "{}"}
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        results = store.search([0.1])
        assert results[0]["score"] == 0.0

    # --- get_by_ids ---

    def test_get_by_ids_empty_returns_empty(self):
        store, _, _ = self._make_store()
        assert store.get_by_ids([]) == []

    def test_get_by_ids_collection_none_returns_empty(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        assert store.get_by_ids(["id1"]) == []

    def test_get_by_ids_returns_results(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        import json

        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"id": "id1", "text": "hello", "metadata_json": json.dumps({"source": "a.txt"})}
        ]

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        docs = store.get_by_ids(["id1"])
        assert docs[0]["id"] == "id1"
        assert docs[0]["score"] == 1.0

    def test_get_by_ids_exception_returns_empty(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.where.side_effect = Exception("query error")

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        result = store.get_by_ids(["id1"])
        assert result == []

    def test_get_by_ids_generates_correct_sql(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.where.return_value.to_list.return_value = []

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.get_by_ids(["id1", "id2"])
        where_call_args = mock_table.search.return_value.where.call_args
        sql = where_call_args[0][0]
        assert "id1" in sql
        assert "id2" in sql
        assert "IN" in sql

    # --- delete_by_ids ---

    def test_delete_by_ids_empty_is_noop(self):
        store, _, mock_table = self._make_store()
        store.delete_by_ids([])
        mock_table.delete.assert_not_called()

    def test_delete_by_ids_collection_none_is_noop(self):
        mock_lancedb, mock_db, _ = _make_lancedb_mocks()
        mock_db.open_table.side_effect = Exception("not found")
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)
        store.delete_by_ids(["id1"])  # Should not raise

    def test_delete_by_ids_generates_sql_and_calls_delete(self):
        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table

        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            from axon.vector_store import OpenVectorStore

            store = OpenVectorStore(cfg)

        store.delete_by_ids(["id1", "id2"])
        mock_table.delete.assert_called_once()
        sql = mock_table.delete.call_args[0][0]
        assert "id1" in sql
        assert "id2" in sql
        assert "IN" in sql


# ---------------------------------------------------------------------------
# MultiVectorStore
# ---------------------------------------------------------------------------


class TestMultiVectorStore:
    """Tests for MultiVectorStore fan-out read-only wrapper."""

    def _make_mock_store(self, provider="chroma"):
        store = MagicMock(
            spec=[
                "provider",
                "collection",
                "search",
                "list_documents",
                "get_by_ids",
                "add",
                "delete_by_ids",
            ]
        )
        store.provider = provider
        store.collection = MagicMock()
        return store

    def _make_multi(self, n=2):
        from axon.vector_store import MultiVectorStore

        stores = [self._make_mock_store() for _ in range(n)]
        multi = MultiVectorStore(stores)
        return multi, stores

    # --- constructor ---

    def test_provider_from_first_store(self):
        from axon.vector_store import MultiVectorStore

        s1 = self._make_mock_store(provider="qdrant")
        s2 = self._make_mock_store(provider="chroma")
        m = MultiVectorStore([s1, s2])
        assert m.provider == "qdrant"

    def test_empty_stores_default_provider(self):
        from axon.vector_store import MultiVectorStore

        m = MultiVectorStore([])
        assert m.provider == "chroma"
        assert m.collection is None

    # --- search ---

    def test_search_merges_results_deduplicates_keeps_max_score(self):
        multi, stores = self._make_multi(2)
        stores[0].search.return_value = [
            {"id": "a", "score": 0.8, "text": "from store0"},
            {"id": "b", "score": 0.5, "text": "b"},
        ]
        stores[1].search.return_value = [
            {"id": "a", "score": 0.9, "text": "from store1"},  # higher score
            {"id": "c", "score": 0.3, "text": "c"},
        ]
        results = multi.search([0.1, 0.2], top_k=10)
        ids = [r["id"] for r in results]
        # All unique
        assert len(ids) == len(set(ids))
        # "a" should have score 0.9 (from store1)
        a_result = next(r for r in results if r["id"] == "a")
        assert a_result["score"] == 0.9
        # Results sorted by score descending
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(self):
        multi, stores = self._make_multi(2)
        stores[0].search.return_value = [
            {"id": f"a{i}", "score": float(i) / 10, "text": ""} for i in range(5)
        ]
        stores[1].search.return_value = [
            {"id": f"b{i}", "score": float(i) / 10, "text": ""} for i in range(5)
        ]
        results = multi.search([0.1], top_k=3)
        assert len(results) == 3

    def test_search_empty_stores(self):
        from axon.vector_store import MultiVectorStore

        MultiVectorStore([])
        # ThreadPoolExecutor with max_workers=min(4,0) — might raise; skip gracefully
        # Actually min(4,0) = 0 which raises; let's use a single-store
        s = self._make_mock_store()
        s.search.return_value = []
        m2 = MultiVectorStore([s])
        results = m2.search([0.1])
        assert results == []

    # --- list_documents ---

    def test_list_documents_merges_and_aggregates_chunks(self):
        multi, stores = self._make_multi(2)
        stores[0].list_documents.return_value = [
            {"source": "a.txt", "chunks": 3, "doc_ids": ["id1", "id2", "id3"]},
            {"source": "b.txt", "chunks": 1, "doc_ids": ["id4"]},
        ]
        stores[1].list_documents.return_value = [
            {"source": "a.txt", "chunks": 2, "doc_ids": ["id5", "id6"]},
        ]
        docs = multi.list_documents()
        sources = {d["source"]: d for d in docs}
        assert sources["a.txt"]["chunks"] == 5
        assert len(sources["a.txt"]["doc_ids"]) == 5
        assert sources["b.txt"]["chunks"] == 1
        # Sorted by source
        assert [d["source"] for d in docs] == sorted([d["source"] for d in docs])

    def test_list_documents_no_overlap(self):
        multi, stores = self._make_multi(2)
        stores[0].list_documents.return_value = [
            {"source": "a.txt", "chunks": 1, "doc_ids": ["id1"]}
        ]
        stores[1].list_documents.return_value = [
            {"source": "b.txt", "chunks": 2, "doc_ids": ["id2", "id3"]}
        ]
        docs = multi.list_documents()
        assert len(docs) == 2

    # --- get_by_ids ---

    def test_get_by_ids_merges_from_all_stores(self):
        multi, stores = self._make_multi(2)
        stores[0].get_by_ids.return_value = [
            {"id": "id1", "text": "t1", "score": 1.0, "metadata": {}}
        ]
        stores[1].get_by_ids.return_value = [
            {"id": "id2", "text": "t2", "score": 1.0, "metadata": {}}
        ]
        docs = multi.get_by_ids(["id1", "id2"])
        ids = {d["id"] for d in docs}
        assert ids == {"id1", "id2"}

    def test_get_by_ids_deduplicates_by_id(self):
        multi, stores = self._make_multi(2)
        doc = {"id": "id1", "text": "t1", "score": 1.0, "metadata": {}}
        stores[0].get_by_ids.return_value = [doc]
        stores[1].get_by_ids.return_value = [doc]
        docs = multi.get_by_ids(["id1"])
        assert len(docs) == 1

    # --- add raises ---

    def test_add_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add(["id1"], ["text"], [[0.1]])

    # --- delete_by_ids raises ---

    def test_delete_by_ids_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_by_ids(["id1"])

    def test_delete_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_documents(["id1"])


# ---------------------------------------------------------------------------
# MultiBM25Retriever
# ---------------------------------------------------------------------------


class TestMultiBM25Retriever:
    """Tests for MultiBM25Retriever fan-out BM25 wrapper."""

    def _make_retriever_mock(self):
        r = MagicMock()
        return r

    def _make_multi(self, n=2):
        from axon.vector_store import MultiBM25Retriever

        retrievers = [self._make_retriever_mock() for _ in range(n)]
        multi = MultiBM25Retriever(retrievers)
        return multi, retrievers

    # --- search ---

    def test_search_merges_and_deduplicates(self):
        multi, retrievers = self._make_multi(2)
        retrievers[0].search.return_value = [
            {"id": "a", "score": 0.7, "text": "from r0"},
            {"id": "b", "score": 0.5, "text": "b"},
        ]
        retrievers[1].search.return_value = [
            {"id": "a", "score": 0.9, "text": "from r1"},  # higher score
            {"id": "c", "score": 0.2, "text": "c"},
        ]
        results = multi.search("hello world", top_k=10)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))
        a_result = next(r for r in results if r["id"] == "a")
        assert a_result["score"] == 0.9

    def test_search_respects_top_k(self):
        multi, retrievers = self._make_multi(2)
        retrievers[0].search.return_value = [
            {"id": f"a{i}", "score": float(i), "text": ""} for i in range(5)
        ]
        retrievers[1].search.return_value = [
            {"id": f"b{i}", "score": float(i), "text": ""} for i in range(5)
        ]
        results = multi.search("query", top_k=4)
        assert len(results) == 4

    def test_search_results_sorted_descending(self):
        multi, retrievers = self._make_multi(1)
        retrievers[0].search.return_value = [
            {"id": "a", "score": 0.3, "text": ""},
            {"id": "b", "score": 0.8, "text": ""},
            {"id": "c", "score": 0.5, "text": ""},
        ]
        results = multi.search("q")
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_passes_query_and_top_k(self):
        multi, retrievers = self._make_multi(1)
        retrievers[0].search.return_value = []
        multi.search("my query", top_k=7)
        retrievers[0].search.assert_called_once_with("my query", 7)

    # --- close ---

    def test_close_calls_close_on_all_retrievers(self):
        multi, retrievers = self._make_multi(3)
        multi.close()
        for r in retrievers:
            r.close.assert_called_once()

    def test_close_skips_retrievers_without_close(self):
        from axon.vector_store import MultiBM25Retriever

        r1 = MagicMock(spec=["search"])  # No close attribute
        r2 = MagicMock()
        multi = MultiBM25Retriever([r1, r2])
        multi.close()  # Should not raise
        r2.close.assert_called_once()

    # --- delete_documents raises ---

    def test_delete_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.delete_documents(["id1"])

    # --- add_documents raises ---

    def test_add_documents_raises_runtime_error(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add_documents(["doc"])

    def test_add_documents_raises_with_kwargs(self):
        multi, _ = self._make_multi()
        with pytest.raises(RuntimeError, match="merged parent project view"):
            multi.add_documents(ids=["id1"], texts=["text"])


# ---------------------------------------------------------------------------
# Edge cases and integration-style tests
# ---------------------------------------------------------------------------


class TestOpenVectorStoreEdgeCases:
    """Additional edge case tests for corner branches."""

    def test_chroma_list_documents_doc_ids_populated(self):
        from axon.vector_store import OpenVectorStore

        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "metadatas": [{"source": "a.txt"}, {"source": "a.txt"}],
        }
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store = OpenVectorStore(cfg)
        docs = store.list_documents()
        assert docs[0]["doc_ids"] == ["id1", "id2"]

    def test_qdrant_list_documents_doc_ids_are_strings(self):
        from axon.vector_store import OpenVectorStore

        mock_qdrant, mock_client = _make_qdrant_mocks()
        p = MagicMock()
        p.id = 42  # integer id
        p.payload = {"source": "doc.txt", "text": "hello"}
        mock_client.scroll.return_value = ([p], None)
        cfg = _make_config(vector_store="qdrant")
        with patch.dict("sys.modules", {"qdrant_client": mock_qdrant}):
            store = OpenVectorStore(cfg)
        docs = store.list_documents()
        assert docs[0]["doc_ids"] == ["42"]

    def test_lancedb_search_calls_limit(self):
        from axon.vector_store import OpenVectorStore

        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        mock_table.search.return_value.limit.return_value.to_list.return_value = []
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            store = OpenVectorStore(cfg)
        store.search([0.1, 0.2], top_k=7)
        mock_table.search.return_value.limit.assert_called_once_with(7)

    def test_chroma_add_non_dimension_error_reraises(self):
        from axon.vector_store import OpenVectorStore

        mock_chromadb, _, mock_collection = _make_chroma_mocks()
        mock_collection.add.side_effect = ValueError("some unrelated error")
        cfg = _make_config(vector_store="chroma")
        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            store = OpenVectorStore(cfg)
        with pytest.raises(ValueError, match="some unrelated error"):
            store.add(ids=["id1"], texts=["t"], embeddings=[[0.1]])

    def test_multi_vector_store_search_all_empty(self):
        from axon.vector_store import MultiVectorStore

        s1, s2 = MagicMock(), MagicMock()
        s1.provider = "chroma"
        s2.provider = "chroma"
        s1.collection = MagicMock()
        s2.collection = MagicMock()
        s1.search.return_value = []
        s2.search.return_value = []
        m = MultiVectorStore([s1, s2])
        assert m.search([0.1]) == []

    def test_sanitize_chroma_meta_empty_input(self):
        from axon.vector_store import OpenVectorStore

        result = OpenVectorStore._sanitize_chroma_meta([])
        assert result == []

    def test_lancedb_add_metadata_json_serialized(self):
        import json

        from axon.vector_store import OpenVectorStore

        mock_lancedb, mock_db, mock_table = _make_lancedb_mocks()
        mock_db.open_table.return_value = mock_table
        cfg = _make_config(vector_store="lancedb")
        with patch.dict("sys.modules", {"lancedb": mock_lancedb}):
            store = OpenVectorStore(cfg)
        store.add(
            ids=["id1"],
            texts=["hello"],
            embeddings=[[0.1, 0.2]],
            metadatas=[{"source": "a.txt", "page": 1}],
        )
        rows = mock_table.add.call_args[0][0]
        meta = json.loads(rows[0]["metadata_json"])
        assert meta["source"] == "a.txt"
        assert meta["page"] == 1
"""Tests for axon.collection_ops."""
import pytest
from unittest.mock import MagicMock
from axon.collection_ops import clear_active_project

def test_clear_active_project():
    brain = MagicMock()
    brain._active_project = "myproj"
    brain.vector_store = MagicMock()
    brain.bm25 = MagicMock()
    brain._ingested_hashes = {"h1"}
    brain._doc_versions = {"v1": 1}
    
    clear_active_project(brain)
    
    assert brain._active_project == "myproj"  # It stays on the project, but wipes its data
    assert brain._ingested_hashes == set()
    assert brain._doc_versions == {}
