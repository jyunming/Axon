from unittest.mock import MagicMock, patch

import pytest

from axon.ext_install import install_vscode_extension


def test_install_vscode_extension_path(capsys):
    with patch("axon.ext_install._find_vsix", return_value="test.vsix"), patch(
        "sys.argv", ["axon-ext", "--path"]
    ):
        install_vscode_extension()
        captured = capsys.readouterr()
        assert "test.vsix" in captured.out


def test_install_vscode_extension_version(capsys):
    mock_vsix = MagicMock()
    mock_vsix.stem = "axon-copilot-0.1.0"
    with patch("axon.ext_install._find_vsix", return_value=mock_vsix), patch(
        "sys.argv", ["axon-ext", "--version"]
    ):
        install_vscode_extension()
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out


def test_install_vscode_extension_no_code(capsys):
    with patch("axon.ext_install._find_vsix", return_value="test.vsix"), patch(
        "axon.ext_install._find_code_cmd", return_value=None
    ), patch("sys.argv", ["axon-ext"]):
        with pytest.raises(SystemExit) as e:
            install_vscode_extension()
        assert e.value.code == 1
        captured = capsys.readouterr()
        assert "ERROR: 'code' command not found" in captured.err
