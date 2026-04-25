"""Phase 2 of the sealed-mount design: ephemeral cache subsystem tests.

Covers ``axon.security.cache``:
- :class:`SealedCache` create + wipe round-trip
- Mixed sealed-and-plaintext source dirs
- Wrong-DEK / tampered-ciphertext rejection (no plaintext leak on failure)
- Capacity check
- PID-based orphan cleanup
- ``is_sealed_file`` magic detection
- Context-manager wipe
- Self-check round-trip

Skips the entire module when the optional ``cryptography`` package isn't
installed (the ``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")

from axon.security.cache import (  # noqa: E402
    CACHE_PREFIX,
    PID_SENTINEL_FILENAME,
    CacheCapacityError,
    SealedCache,
    _self_check,
    cleanup_orphans,
    is_sealed_file,
    list_orphans,
)
from axon.security.crypto import SealedFile, generate_dek, make_aad  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seal_project(sealed_dir: Path, dek: bytes, *, key_id: str, files: dict[str, bytes]) -> None:
    """Write *files* (relpath → plaintext) into *sealed_dir* as AXSL-sealed."""
    sealed_dir.mkdir(parents=True, exist_ok=True)
    for rel, payload in files.items():
        target = sealed_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        SealedFile.write(target, payload, dek, aad=make_aad(key_id, rel))


# ---------------------------------------------------------------------------
# is_sealed_file
# ---------------------------------------------------------------------------


class TestIsSealedFile:
    def test_returns_true_for_sealed_file(self, tmp_path):
        path = tmp_path / "x.bin"
        SealedFile.write(path, b"hi", generate_dek())
        assert is_sealed_file(path) is True

    def test_returns_false_for_plaintext(self, tmp_path):
        path = tmp_path / "y.txt"
        path.write_text("plain", encoding="utf-8")
        assert is_sealed_file(path) is False

    def test_returns_false_for_missing(self, tmp_path):
        assert is_sealed_file(tmp_path / "nope") is False

    def test_returns_false_for_directory(self, tmp_path):
        assert is_sealed_file(tmp_path) is False


# ---------------------------------------------------------------------------
# SealedCache.create — round-trip
# ---------------------------------------------------------------------------


class TestCacheCreate:
    def test_round_trip_single_file(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b'{"id":"x"}'})

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        try:
            assert (cache.path / "meta.json").read_bytes() == b'{"id":"x"}'
        finally:
            cache.wipe()

    def test_round_trip_nested_dirs(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        files = {
            "meta.json": b'{"id":"x"}',
            "vector_store_data/manifest.json": b'{"d":768}',
            "vector_store_data/seg-00000001.bin": b"\x01" * 1024,
            "bm25_index/.bm25_log.jsonl": b'{"seq":1}\n',
        }
        _seal_project(sealed, dek, key_id="sk_a", files=files)

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        try:
            for rel, payload in files.items():
                assert (cache.path / rel).read_bytes() == payload
        finally:
            cache.wipe()

    def test_passthrough_for_non_sealed_files(self, tmp_path):
        """Files without an AXSL header are copied as-is — meant for
        ``version.json`` which deliberately stays plaintext."""
        sealed = tmp_path / "proj"
        dek = generate_dek()
        sealed.mkdir()
        # Plaintext version marker — no AXSL header.
        (sealed / "version.json").write_text('{"seq":42}', encoding="utf-8")
        # And one sealed file alongside it.
        SealedFile.write(
            sealed / "meta.json",
            b'{"id":"x"}',
            dek,
            aad=make_aad("sk_a", "meta.json"),
        )

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        try:
            # Plaintext file copied verbatim.
            assert (cache.path / "version.json").read_text(encoding="utf-8") == '{"seq":42}'
            # Sealed file decrypted.
            assert (cache.path / "meta.json").read_bytes() == b'{"id":"x"}'
        finally:
            cache.wipe()

    def test_pid_sentinel_written(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        try:
            sentinel = cache.path / PID_SENTINEL_FILENAME
            assert sentinel.exists()
            assert int(sentinel.read_text(encoding="utf-8")) == os.getpid()
        finally:
            cache.wipe()

    def test_cache_dir_uses_expected_prefix(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        try:
            assert cache.path.name.startswith(CACHE_PREFIX)
            assert cache.path.parent == tmp_path
        finally:
            cache.wipe()

    def test_missing_sealed_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SealedCache.create(tmp_path / "does-not-exist", generate_dek(), key_id="sk_a")


# ---------------------------------------------------------------------------
# SealedCache.create — failure paths
# ---------------------------------------------------------------------------


class TestCacheCreateFailures:
    def test_wrong_dek_raises_invalid_tag_and_wipes_partial(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        sealed = tmp_path / "proj"
        right_dek = generate_dek()
        wrong_dek = generate_dek()
        _seal_project(sealed, right_dek, key_id="sk_a", files={"meta.json": b"{}"})

        # Snapshot existing temp dir contents so we can detect a partial.
        before = {p.name for p in tmp_path.iterdir()}
        with pytest.raises(InvalidTag):
            SealedCache.create(sealed, wrong_dek, key_id="sk_a", cache_root=tmp_path)
        after = {p.name for p in tmp_path.iterdir()}
        # No new axon-sealed-* dir survived the failure.
        new_caches = [n for n in (after - before) if n.startswith(CACHE_PREFIX)]
        assert new_caches == []

    def test_wrong_aad_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_orig", files={"meta.json": b"{}"})

        # Different key_id → AAD mismatch → InvalidTag.
        with pytest.raises(InvalidTag):
            SealedCache.create(sealed, dek, key_id="sk_other", cache_root=tmp_path)

    def test_capacity_error_when_disk_too_small(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"x" * 1000})

        # Mock free space to a value below the 1.1× headroom threshold.
        from collections import namedtuple

        Usage = namedtuple("Usage", "total used free")

        with patch(
            "axon.security.cache.shutil.disk_usage",
            return_value=Usage(total=1_000_000, used=999_000, free=10),
        ):
            with pytest.raises(CacheCapacityError, match="bytes free"):
                SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)


# ---------------------------------------------------------------------------
# SealedCache.wipe + context manager
# ---------------------------------------------------------------------------


class TestCacheWipe:
    def test_wipe_removes_cache_dir(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        cache_path = cache.path
        assert cache_path.exists()
        cache.wipe()
        assert not cache_path.exists()

    def test_wipe_is_idempotent(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        cache.wipe()
        # Second wipe must NOT raise.
        cache.wipe()

    def test_wipe_removes_nested_files(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(
            sealed,
            dek,
            key_id="sk_a",
            files={
                "meta.json": b"{}",
                "vector_store_data/manifest.json": b"{}",
                "vector_store_data/seg-00000001.bin": b"\x00" * 4096,
                "bm25_index/.bm25_log.jsonl": b"{}\n",
            },
        )

        cache = SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path)
        cache_path = cache.path
        assert (cache_path / "vector_store_data" / "seg-00000001.bin").exists()
        cache.wipe()
        # Whole tree gone — including the nested subdirs.
        assert not cache_path.exists()

    def test_context_manager_wipes_on_exit(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        with SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path) as cache:
            cache_path = cache.path
            assert cache_path.exists()
        assert not cache_path.exists()

    def test_context_manager_wipes_even_on_exception(self, tmp_path):
        sealed = tmp_path / "proj"
        dek = generate_dek()
        _seal_project(sealed, dek, key_id="sk_a", files={"meta.json": b"{}"})

        cache_path_holder: list[Path] = []
        with pytest.raises(RuntimeError):
            with SealedCache.create(sealed, dek, key_id="sk_a", cache_root=tmp_path) as cache:
                cache_path_holder.append(cache.path)
                raise RuntimeError("simulated failure mid-use")
        assert not cache_path_holder[0].exists()


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------


class TestOrphanCleanup:
    def _make_orphan_with_pid(self, root: Path, *, pid: int, with_files: bool = True) -> Path:
        """Hand-craft a cache-shaped dir with the given PID sentinel."""
        import tempfile as _tempfile

        d = Path(_tempfile.mkdtemp(prefix=CACHE_PREFIX, dir=str(root)))
        (d / PID_SENTINEL_FILENAME).write_text(str(pid), encoding="utf-8")
        if with_files:
            (d / "junk.bin").write_bytes(b"residue" * 100)
        return d

    def test_dead_pid_listed_as_orphan(self, tmp_path):
        # PID 0 is never a valid live process.
        orphan = self._make_orphan_with_pid(tmp_path, pid=0)
        orphans = list_orphans(cache_root=tmp_path)
        assert orphan in orphans

    def test_alive_pid_not_listed(self, tmp_path):
        # The current process IS alive.
        not_orphan = self._make_orphan_with_pid(tmp_path, pid=os.getpid())
        orphans = list_orphans(cache_root=tmp_path)
        assert not_orphan not in orphans

    def test_missing_pid_file_treated_as_orphan(self, tmp_path):
        """Defensive: if the sentinel is gone or unreadable, wipe.
        A real active cache always writes its sentinel."""
        import tempfile as _tempfile

        d = Path(_tempfile.mkdtemp(prefix=CACHE_PREFIX, dir=str(tmp_path)))
        (d / "stuff.bin").write_bytes(b"x")
        # No PID file.
        orphans = list_orphans(cache_root=tmp_path)
        assert d in orphans

    def test_unparseable_pid_treated_as_orphan(self, tmp_path):
        import tempfile as _tempfile

        d = Path(_tempfile.mkdtemp(prefix=CACHE_PREFIX, dir=str(tmp_path)))
        (d / PID_SENTINEL_FILENAME).write_text("not a number", encoding="utf-8")
        orphans = list_orphans(cache_root=tmp_path)
        assert d in orphans

    def test_non_axon_dirs_ignored(self, tmp_path):
        unrelated = tmp_path / "some-other-temp"
        unrelated.mkdir()
        (unrelated / PID_SENTINEL_FILENAME).write_text("0", encoding="utf-8")
        assert list_orphans(cache_root=tmp_path) == []

    def test_cleanup_orphans_wipes_dead_caches(self, tmp_path):
        dead = self._make_orphan_with_pid(tmp_path, pid=0)
        alive = self._make_orphan_with_pid(tmp_path, pid=os.getpid())
        wiped = cleanup_orphans(cache_root=tmp_path)
        assert wiped == 1
        assert not dead.exists()
        assert alive.exists()  # left alone

    def test_cleanup_orphans_returns_zero_when_none(self, tmp_path):
        # Empty temp dir.
        assert cleanup_orphans(cache_root=tmp_path) == 0

    def test_cleanup_orphans_never_raises(self, tmp_path):
        # Even if the temp dir doesn't exist, return 0 cleanly.
        assert cleanup_orphans(cache_root=tmp_path / "does-not-exist") == 0


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------


class TestSelfCheck:
    def test_self_check_ok_on_healthy_install(self):
        result = _self_check()
        assert result["ok"] is True
        assert "round-trip OK" in result["details"]
