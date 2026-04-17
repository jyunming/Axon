from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from axon.vector_store import OpenVectorStore
from axon.config import AxonConfig


def test_add_unbalanced_inputs_raises():
    """Verify behavior with mismatched input lengths."""
    cfg = AxonConfig()
    # Prevent store initialization (avoid heavy deps)
    with patch.object(OpenVectorStore, "_init_store", lambda self: None):
        store = OpenVectorStore(cfg)
        store.provider = "chroma"
        store.collection = MagicMock()
        with pytest.raises(ValueError, match="length mismatch"):
            store.add(["id1", "id2"], ["text1"], [[0.1]])


def test_add_metadatas_mismatch_raises():
    """Verify metadatas length mismatch raises."""
    cfg = AxonConfig()
    with patch.object(OpenVectorStore, "_init_store", lambda self: None):
        store = OpenVectorStore(cfg)
        store.provider = "chroma"
        store.collection = MagicMock()
        with pytest.raises(ValueError, match="metadatas length mismatch"):
            store.add(["id1"], ["text1"], [[0.1]], metadatas=[{}, {}])
