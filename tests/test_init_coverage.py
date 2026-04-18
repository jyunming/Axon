from unittest.mock import patch

import pytest

import axon


def test_init_lazy_imports():
    # Trigger lazy imports in axon/__init__.py
    assert axon.AxonConfig is not None
    assert axon.AxonBrain is not None
    assert axon.OpenEmbedding is not None
    assert axon.OpenLLM is not None
    assert axon.OpenVectorStore is not None
    assert axon.__version__ is not None


def test_init_invalid_attr():
    with pytest.raises(AttributeError):
        _ = axon.NonExistentAttr


def test_init_version_not_found():
    from importlib.metadata import PackageNotFoundError

    with patch("importlib.metadata.version", side_effect=PackageNotFoundError):
        # We need to reload the module to trigger the top-level code again
        import importlib

        import axon

        importlib.reload(axon)
        assert axon.__version__ == "0.0.0+dev"


def test_access_exception_handling():
    from axon.access import check_write_allowed

    with patch("axon.projects.get_maintenance_state", side_effect=RuntimeError("test error")):
        # Should not raise exception because it's caught and ignored
        check_write_allowed("not-default", "test", read_only_scope=False, is_mounted=False)
