import os
from unittest.mock import MagicMock, patch

from axon.loaders import DirectoryLoader, ImageLoader, LegacyOfficeLoader, PPTXLoader


def test_pptx_loader_mock():
    mock_pptx = MagicMock()
    mock_prs = MagicMock()
    mock_slide = MagicMock()
    mock_shape = MagicMock()
    mock_shape.text = "Hello PPTX"
    mock_slide.shapes = [mock_shape]
    mock_prs.slides = [mock_slide]
    mock_pptx.Presentation.return_value = mock_prs

    with patch.dict("sys.modules", {"pptx": mock_pptx}):
        loader = PPTXLoader()
        # Create a dummy file for size check
        with open("test.pptx", "w") as f:
            f.write("dummy")
        try:
            docs = loader.load("test.pptx")
            assert len(docs) == 1
            assert docs[0]["text"] == "Hello PPTX"
            assert docs[0]["metadata"]["type"] == "pptx"
        finally:
            if os.path.exists("test.pptx"):
                os.remove("test.pptx")


def test_legacy_office_loader_mock():
    mock_textract = MagicMock()
    mock_textract.process.return_value = b"Legacy Content"
    with patch.dict("sys.modules", {"textract": mock_textract}):
        loader = LegacyOfficeLoader()
        with open("test.doc", "w") as f:
            f.write("dummy")
        try:
            docs = loader.load("test.doc")
            assert len(docs) == 1
            assert docs[0]["text"] == "Legacy Content"
            assert docs[0]["metadata"]["type"] == "legacy_office"
        finally:
            if os.path.exists("test.doc"):
                os.remove("test.doc")


def test_image_loader_jpg_mock():
    mock_img = MagicMock()
    mock_img.convert.return_value = mock_img

    with patch("PIL.Image.open", return_value=mock_img):
        with patch("ollama.generate", return_value={"response": "A sunny day"}):
            loader = ImageLoader()
            with open("test.jpg", "w") as f:
                f.write("dummy")
            try:
                docs = loader.load("test.jpg")
                assert len(docs) == 1
                assert "A sunny day" in docs[0]["text"]
                assert docs[0]["metadata"]["type"] == "image"
                assert docs[0]["metadata"]["format"] == "jpg"
            finally:
                if os.path.exists("test.jpg"):
                    os.remove("test.jpg")


def test_directory_loader_registers_new_extensions():
    loader = DirectoryLoader()
    assert ".pptx" in loader.loaders
    assert ".doc" in loader.loaders
    assert ".ppt" in loader.loaders
    assert ".jpg" in loader.loaders
    assert ".jpeg" in loader.loaders
    assert ".png" in loader.loaders
