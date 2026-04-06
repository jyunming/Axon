import sys
from unittest.mock import patch

import pytest

from axon.cli import main as cli_main


def test_cli_setup_invocation():
    with patch("axon.config_wizard.run_wizard") as mock_wizard:
        with patch.object(sys, "argv", ["axon", "--setup"]):
            with pytest.raises(SystemExit):
                cli_main()
            mock_wizard.assert_called_once()


def test_cli_version_flag(capsys):
    with patch.object(sys, "argv", ["axon", "--version"]):
        with pytest.raises(SystemExit) as e:
            cli_main()
        assert e.value.code == 0
        captured = capsys.readouterr()
        assert "0.1.0" in captured.out


def test_cli_help_flag(capsys):
    with patch.object(sys, "argv", ["axon", "--help"]):
        with pytest.raises(SystemExit) as e:
            cli_main()
        assert e.value.code == 0
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()


def test_cli_invalid_command(capsys):
    with patch.object(sys, "argv", ["axon", "--invalid-flag-xyz"]):
        with pytest.raises(SystemExit):
            cli_main()
