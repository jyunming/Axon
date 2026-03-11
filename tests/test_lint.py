import subprocess
import sys


def test_ruff_linting():
    """Verify that ruff check passes across src/ and tests/."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"Ruff found linting issues:\n{result.stdout}\n{result.stderr}"


def test_black_formatting():
    """Verify that black --check passes (no formatting changes needed)."""
    result = subprocess.run(
        [sys.executable, "-m", "black", "--check", "src/", "tests/"], capture_output=True, text=True
    )
    assert (
        result.returncode == 0
    ), f"Black found formatting issues:\n{result.stdout}\n{result.stderr}"
