"""Tests for v0.4.0 PR I — ``scripts/render_index.py``.

Coverage:
- Reads version from a real ``Cargo.toml`` shape.
- Substitutes every ``{{AXON_VERSION}}`` in the template.
- ``--check`` mode: in-sync exit 0, out-of-sync exit 1, no write.
- ``--version`` override skips Cargo.toml read.
- Real repo's template/output stays in sync (drift guard).

These tests run via subprocess so they exercise the actual entrypoint
the bump-and-release workflow calls — not just an importable function.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "render_index.py"


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Invoke render_index.py at ``cwd/scripts/render_index.py``. The
    script uses ``__file__`` to find the repo root, so we MUST run the
    copy inside the seeded temp dir, not the source repo's script."""
    script = cwd / "scripts" / "render_index.py"
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _seed_repo(tmp_path: Path, version: str, template_text: str) -> Path:
    """Create the minimal repo shape render_index.py expects."""
    cargo_dir = tmp_path / "src" / "axon"
    cargo_dir.mkdir(parents=True)
    (cargo_dir / "Cargo.toml").write_text(
        f'[package]\nname = "axon"\nversion = "{version}"\n',
        encoding="utf-8",
    )
    template = tmp_path / "index.template.html"
    template.write_text(template_text, encoding="utf-8")
    # The script lives under scripts/ relative to repo root; replicate.
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    shutil.copy2(SCRIPT, scripts_dir / "render_index.py")
    return tmp_path


def test_render_substitutes_all_placeholders(tmp_path):
    repo = _seed_repo(
        tmp_path,
        "1.2.3",
        "Hero v{{AXON_VERSION}} terminal {{AXON_VERSION}} sample v{{AXON_VERSION}}\n",
    )
    result = _run(["--expected-placeholders", "3"], cwd=repo)
    assert result.returncode == 0, result.stderr
    rendered = (repo / "index.html").read_text(encoding="utf-8")
    assert "{{AXON_VERSION}}" not in rendered
    assert "Hero v1.2.3 terminal 1.2.3 sample v1.2.3" in rendered


def test_render_uses_version_override(tmp_path):
    repo = _seed_repo(tmp_path, "1.2.3", "v{{AXON_VERSION}}\n")
    result = _run(["--version", "9.9.9", "--expected-placeholders", "1"], cwd=repo)
    assert result.returncode == 0, result.stderr
    rendered = (repo / "index.html").read_text(encoding="utf-8")
    assert "v9.9.9" in rendered
    assert "1.2.3" not in rendered  # Cargo version was ignored


def test_render_count_in_stdout(tmp_path):
    repo = _seed_repo(
        tmp_path,
        "0.5.0",
        "{{AXON_VERSION}} {{AXON_VERSION}} {{AXON_VERSION}}\n",
    )
    result = _run(["--expected-placeholders", "3"], cwd=repo)
    assert result.returncode == 0
    assert "3 substitution(s)" in result.stdout
    assert "version=0.5.0" in result.stdout


def test_check_passes_when_in_sync(tmp_path):
    repo = _seed_repo(tmp_path, "0.5.0", "v{{AXON_VERSION}}\n")
    # First render to produce a synced output
    _run(["--expected-placeholders", "1"], cwd=repo)
    result = _run(["--check", "--expected-placeholders", "1"], cwd=repo)
    assert result.returncode == 0, result.stderr
    assert "in sync" in result.stdout


def test_check_fails_when_drifted(tmp_path):
    repo = _seed_repo(tmp_path, "0.5.0", "v{{AXON_VERSION}}\n")
    _run(["--expected-placeholders", "1"], cwd=repo)
    # Hand-edit index.html to simulate drift
    output = repo / "index.html"
    output.write_text("v9.9.9\n", encoding="utf-8")
    result = _run(["--check", "--expected-placeholders", "1"], cwd=repo)
    assert result.returncode == 1
    assert "OUT OF SYNC" in result.stdout


def test_check_does_not_write(tmp_path):
    repo = _seed_repo(tmp_path, "0.5.0", "v{{AXON_VERSION}}\n")
    output = repo / "index.html"
    output.write_text("manual edit\n", encoding="utf-8")
    before = output.read_text(encoding="utf-8")
    _run(["--check", "--expected-placeholders", "1"], cwd=repo)
    assert output.read_text(encoding="utf-8") == before, "--check must never modify index.html"


def test_template_without_placeholder_errors(tmp_path):
    repo = _seed_repo(tmp_path, "0.5.0", "no placeholders here\n")
    result = _run(["--expected-placeholders", "1"], cwd=repo)
    assert result.returncode != 0
    assert "no {{AXON_VERSION}} placeholder" in (result.stderr + result.stdout)


def test_missing_template_errors(tmp_path):
    repo = _seed_repo(tmp_path, "0.5.0", "v{{AXON_VERSION}}\n")
    (repo / "index.template.html").unlink()
    result = _run(["--expected-placeholders", "1"], cwd=repo)
    assert result.returncode != 0
    assert "Missing template" in (result.stderr + result.stdout)


def test_placeholder_count_mismatch_errors(tmp_path):
    """Copilot finding on PR #110: a literal version hardcoded into
    one of the templated slots must trip the count check, not silently
    pass --check."""
    repo = _seed_repo(
        tmp_path,
        "0.5.0",
        "v{{AXON_VERSION}} hardcoded v9.9.9 here\n",
    )
    # Template has 1 placeholder but we declare 5 expected
    result = _run(["--expected-placeholders", "5"], cwd=repo)
    assert result.returncode != 0
    err = result.stderr + result.stdout
    assert "expected 5" in err
    assert "EXPECTED_PLACEHOLDER_COUNT" in err


# ---------------------------------------------------------------------------
# Real-repo drift guard
# ---------------------------------------------------------------------------


def test_real_repo_index_in_sync():
    """The committed ``index.html`` MUST match what
    ``render_index.py`` would produce given the current Cargo.toml
    version. This catches a hand-edit-without-template-update
    regression in CI.

    Asserts (not skips) when ``index.template.html`` is missing: the
    template is now load-bearing for the release workflow, so its
    absence is a regression to surface, not silently tolerate
    (Copilot finding on PR #110).
    """
    template = REPO_ROOT / "index.template.html"
    assert (
        template.exists()
    ), f"missing source-of-truth: {template.relative_to(REPO_ROOT)} (see PR I)"
    result = _run(["--check"], cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"index.html drifted from index.template.html.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
