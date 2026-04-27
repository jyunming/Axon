"""Phase 2 part 4: sealed-project mount tests.

Covers ``axon.security.mount`` (the materialise/release glue) and the
``AxonBrain.switch_project`` / ``close()`` / ``__init__`` lifecycle
hooks that route sealed projects through an ephemeral plaintext cache.

Skips when the optional ``[sealed]`` extra is not installed.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

from axon.security import SecurityError  # noqa: E402
from axon.security.cache import (  # noqa: E402
    CACHE_PREFIX,
    PID_SENTINEL_FILENAME,
    cleanup_orphans,
    is_sealed_file,
)
from axon.security.master import bootstrap_store, lock_store  # noqa: E402
from axon.security.mount import materialize_for_read, release_cache  # noqa: E402
from axon.security.seal import project_seal  # noqa: E402

# ---------------------------------------------------------------------------
# In-memory keyring fixture (mirrors the other sealed-* test files)
# ---------------------------------------------------------------------------


class _InMemoryKeyring:
    priority = 1

    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, secret):
        self._store[(service, username)] = secret

    def get_password(self, service, username):
        return self._store.get((service, username))

    def delete_password(self, service, username):
        import keyring.errors

        if (service, username) not in self._store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self._store[(service, username)]


@pytest.fixture
def kr_backend():
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


@pytest.fixture
def user_dir(tmp_path):
    ud = tmp_path / "alice"
    ud.mkdir()
    return ud


def _populate_and_seal(user_dir: Path, project: str = "research") -> Path:
    """Build a representative project, seal it, return its directory."""
    proj = user_dir / project
    (proj / "bm25_index").mkdir(parents=True)
    (proj / "vector_store_data").mkdir(parents=True)
    (proj / ".security").mkdir(parents=True)
    (proj / "meta.json").write_text('{"project_id":"p1","name":"research"}', encoding="utf-8")
    (proj / "version.json").write_text('{"seq":1}', encoding="utf-8")  # passthrough
    (proj / "bm25_index" / ".bm25_log.jsonl").write_text('{"id":"d1"}\n', encoding="utf-8")
    (proj / "vector_store_data" / "manifest.json").write_text('{"d":768}', encoding="utf-8")
    (proj / "vector_store_data" / "seg-00000001.bin").write_bytes(b"\xab" * 4096)

    bootstrap_store(user_dir, "test-pass-ok")
    project_seal(project, user_dir)
    return proj


# ---------------------------------------------------------------------------
# materialize_for_read — happy path + decryption fidelity
# ---------------------------------------------------------------------------


class TestMaterializeForReadHappyPath:
    def test_decrypts_into_temp_cache(self, kr_backend, user_dir, tmp_path):
        proj = _populate_and_seal(user_dir)
        cache_root = tmp_path / "cache_root"
        cache_root.mkdir()

        cache = materialize_for_read(proj, user_dir, cache_root=cache_root)
        try:
            # Cache dir lives under our cache_root override and uses
            # the documented prefix.
            assert cache.path.parent == cache_root
            assert cache.path.name.startswith(CACHE_PREFIX)
            # PID sentinel exists.
            assert (cache.path / PID_SENTINEL_FILENAME).read_text(encoding="utf-8") == str(
                os.getpid()
            )
            # Plaintext content matches what we put in pre-seal.
            assert (
                cache.path / "vector_store_data" / "seg-00000001.bin"
            ).read_bytes() == b"\xab" * 4096
            assert (cache.path / "vector_store_data" / "manifest.json").read_text(
                encoding="utf-8"
            ) == '{"d":768}'
            assert (cache.path / "bm25_index" / ".bm25_log.jsonl").read_text(
                encoding="utf-8"
            ) == '{"id":"d1"}\n'
            assert (cache.path / "meta.json").read_text(
                encoding="utf-8"
            ) == '{"project_id":"p1","name":"research"}'
            # Passthrough plaintext file copied as-is.
            assert (cache.path / "version.json").read_text(encoding="utf-8") == '{"seq":1}'
        finally:
            release_cache(cache)
        # release_cache wipes the directory.
        assert not cache.path.exists()

    def test_release_cache_safe_on_none(self):
        # No-op for the convenience wrapper.
        release_cache(None)


# ---------------------------------------------------------------------------
# materialize_for_read — error / refusal paths
# ---------------------------------------------------------------------------


class TestMaterializeForReadErrors:
    def test_refuses_unsealed_project(self, kr_backend, user_dir, tmp_path):
        proj = user_dir / "open_proj"
        proj.mkdir()
        (proj / "meta.json").write_text("{}", encoding="utf-8")
        with pytest.raises(SecurityError, match="non-sealed"):
            materialize_for_read(proj, user_dir, cache_root=tmp_path / "cr")

    def test_refuses_when_locked(self, kr_backend, user_dir, tmp_path):
        proj = _populate_and_seal(user_dir)
        # Lock the store after seal; materialize should refuse.
        lock_store(user_dir)
        with pytest.raises(SecurityError, match="locked"):
            materialize_for_read(proj, user_dir, cache_root=tmp_path / "cr")

    def test_refuses_when_marker_missing_seal_id(self, kr_backend, user_dir, tmp_path, monkeypatch):
        proj = _populate_and_seal(user_dir)

        # Patch read_sealed_marker to return a marker without seal_id.
        from axon.security import mount as _mount

        monkeypatch.setattr(_mount, "read_sealed_marker", lambda _p: {"v": 1, "seal_id": ""})
        with pytest.raises(SecurityError, match="no seal_id"):
            materialize_for_read(proj, user_dir, cache_root=tmp_path / "cr")

    def test_wraps_underlying_failure_as_security_error(
        self, kr_backend, user_dir, tmp_path, monkeypatch
    ):
        proj = _populate_and_seal(user_dir)

        # Force SealedCache.create to raise an arbitrary OSError —
        # mount.py should wrap it into SecurityError.
        from axon.security import mount as _mount

        def _boom(*_a, **_kw):
            raise OSError("disk full")

        monkeypatch.setattr(_mount.SealedCache, "create", classmethod(_boom))
        with pytest.raises(SecurityError, match="disk full"):
            materialize_for_read(proj, user_dir, cache_root=tmp_path / "cr")


# ---------------------------------------------------------------------------
# Round-trip — seal, mount, read, wipe, verify ciphertext on disk unchanged
# ---------------------------------------------------------------------------


class TestSealMountRoundTrip:
    def test_disk_stays_ciphertext_after_mount(self, kr_backend, user_dir, tmp_path):
        proj = _populate_and_seal(user_dir)

        # Confirm pre-mount: content files on disk are AXSL ciphertext.
        assert is_sealed_file(proj / "meta.json")
        assert is_sealed_file(proj / "vector_store_data" / "seg-00000001.bin")

        cache = materialize_for_read(proj, user_dir, cache_root=tmp_path / "cr")
        try:
            # While mounted: cache has plaintext, original is still sealed.
            assert not is_sealed_file(cache.path / "meta.json")
            assert is_sealed_file(proj / "meta.json")  # untouched
        finally:
            release_cache(cache)

        # Post-wipe: original is still sealed.
        assert is_sealed_file(proj / "meta.json")


# ---------------------------------------------------------------------------
# AxonBrain integration — switch_project / close / __init__
# Use targeted attribute checks rather than instantiating a real brain
# (heavy: pulls in embedders + LLM).
# ---------------------------------------------------------------------------


class TestBrainHooksContract:
    """Lightweight contract checks for the brain-side wiring.

    Avoids constructing a full ``AxonBrain`` (loads embeddings, executor,
    config). Instead exercises the private helpers directly on a
    minimal duck-typed object so we can verify the path-routing logic
    without paying the AxonBrain init cost.
    """

    def test_project_is_sealed_returns_false_for_open(self, kr_backend, user_dir):
        from axon.main import AxonBrain

        proj = user_dir / "open_proj"
        proj.mkdir()
        (proj / "meta.json").write_text("{}", encoding="utf-8")
        # Use any object as `self` — the method only consults its arg.
        stub: object = object()
        result = AxonBrain._project_is_sealed(stub, proj)  # type: ignore[arg-type]
        assert result is False

    def test_project_is_sealed_returns_true_after_seal(self, kr_backend, user_dir):
        from axon.main import AxonBrain

        proj = _populate_and_seal(user_dir)
        stub: object = object()
        result = AxonBrain._project_is_sealed(stub, proj)  # type: ignore[arg-type]
        assert result is True


# ---------------------------------------------------------------------------
# Orphan cleanup — survives across "process restarts"
# ---------------------------------------------------------------------------


class TestOrphanCleanupAcrossRestart:
    def test_orphan_with_dead_pid_is_wiped(self, tmp_path, kr_backend, user_dir):
        proj = _populate_and_seal(user_dir)
        cache_root = tmp_path / "cr"
        cache_root.mkdir()

        # Create a "stale" cache then forge a non-existent PID into the
        # sentinel — simulates a crashed previous session.
        cache = materialize_for_read(proj, user_dir, cache_root=cache_root)
        cache_dir = cache.path
        # Forge: pick a PID that is extremely unlikely to be alive.
        # PID 999999999 isn't valid on Linux/Windows.
        (cache_dir / PID_SENTINEL_FILENAME).write_text("999999999", encoding="utf-8")

        # Don't release — let cleanup_orphans find it as if a previous
        # process died holding it.
        wiped = cleanup_orphans(cache_root=cache_root)
        assert wiped >= 1
        assert not cache_dir.exists()

    def test_active_pid_is_left_alone(self, tmp_path, kr_backend, user_dir):
        proj = _populate_and_seal(user_dir)
        cache_root = tmp_path / "cr"
        cache_root.mkdir()

        cache = materialize_for_read(proj, user_dir, cache_root=cache_root)
        try:
            # Sentinel has THIS process's PID — must not be wiped.
            wiped = cleanup_orphans(cache_root=cache_root)
            assert wiped == 0
            assert cache.path.exists()
        finally:
            release_cache(cache)
