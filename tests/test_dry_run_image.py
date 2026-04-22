from __future__ import annotations

from unittest.mock import MagicMock


def test_image_loader_respects_dry_run(tmp_path, monkeypatch):
    """ImageLoader should not call external VLMs when AXON_DRY_RUN is set.

    This test ensures that during --dry-run the loader skips any calls to
    external providers like Ollama and returns a placeholder document.
    """
    monkeypatch.setenv("AXON_DRY_RUN", "1")

    from axon.loaders import ImageLoader

    loader = ImageLoader(ollama_model="llava")

    # Force dependencies to appear present so the loader would normally attempt a VLM call
    loader._pil = MagicMock()
    img_mock = MagicMock()
    loader._pil.open.return_value = img_mock
    img_mock.convert.return_value = img_mock
    img_mock.save.return_value = None

    loader.ollama = MagicMock()
    loader.ollama.generate = MagicMock(side_effect=AssertionError("Should not be called"))

    p = tmp_path / "img.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")

    docs = loader.load(str(p))

    # Ollama.generate must not be invoked
    assert not loader.ollama.generate.called, "Ollama.generate was called during dry-run"
    assert isinstance(docs, list)
    assert docs, "Expected at least one placeholder document during dry-run"
