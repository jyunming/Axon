"""Tests for v0.4.0 Item 4 — metadata leakage hardening.

Coverage:

4a. Hostname → store-scoped UUID node_id
    - ``init_store`` writes a fresh ``node_id`` UUID into ``store_meta.json``.
    - ``get_or_create_node_id`` returns the stored value, mints + persists
      one for legacy stores (migration path).
    - ``version_marker.bump`` accepts ``node_id`` and writes
      ``owner_node_id`` + empty ``owner_host`` (no hostname leak).
    - ``socket`` is no longer imported by ``version_marker``.

4c. Random padding in sealed files (``security.seal_padding_bytes``)
    - ``SealedFile.write`` round-trips with non-zero padding.
    - ``write_stream`` round-trips with non-zero padding.
    - File size grows by ``padding_length`` bytes; header carries the
      length so ``read`` can slice it off.
    - Two writes of the same plaintext with the same key produce
      different on-disk lengths most of the time (defeats the size leak).
    - ``AxonConfig.seal_padding_bytes`` round-trips through YAML.

4b (hashed key_id filenames) is deferred to v0.5.0 — see PR description.
"""
from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# 4a — node_id replaces hostname
# ---------------------------------------------------------------------------


class TestNodeId:
    def test_init_store_writes_node_id(self, tmp_path):
        """A freshly initialised store must carry a UUID-shaped
        ``node_id`` in ``store_meta.json`` — replacement for the
        previous behaviour of letting bumps stamp the OS hostname into
        each project's ``version.json``."""
        from axon.projects import ensure_user_project

        user_dir = tmp_path / "AxonStore" / "alice"
        user_dir.mkdir(parents=True)
        ensure_user_project(user_dir)
        meta = json.loads((user_dir / "store_meta.json").read_text(encoding="utf-8"))
        assert "node_id" in meta
        nid = meta["node_id"]
        # UUID4 string: 8-4-4-4-12 hex digits separated by dashes
        parts = nid.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]

    def test_get_or_create_node_id_returns_existing(self, tmp_path):
        from axon.projects import ensure_user_project, get_or_create_node_id

        user_dir = tmp_path / "AxonStore" / "alice"
        user_dir.mkdir(parents=True)
        ensure_user_project(user_dir)
        first = get_or_create_node_id(user_dir)
        second = get_or_create_node_id(user_dir)
        assert first == second
        assert first  # non-empty

    def test_get_or_create_node_id_migrates_legacy_store(self, tmp_path):
        """A pre-v0.4.0 ``store_meta.json`` without a ``node_id`` field
        must get one minted in-place on first read."""
        from axon.projects import get_or_create_node_id

        user_dir = tmp_path / "AxonStore" / "bob"
        user_dir.mkdir(parents=True)
        legacy = {"store_version": 2, "store_id": "store_legacy"}
        (user_dir / "store_meta.json").write_text(json.dumps(legacy), encoding="utf-8")
        nid = get_or_create_node_id(user_dir)
        assert nid
        # Persisted: a second call returns the same value
        assert get_or_create_node_id(user_dir) == nid
        # And ``store_meta.json`` now has the field
        meta = json.loads((user_dir / "store_meta.json").read_text(encoding="utf-8"))
        assert meta["node_id"] == nid

    def test_get_or_create_node_id_missing_meta_returns_empty(self, tmp_path):
        from axon.projects import get_or_create_node_id

        user_dir = tmp_path / "AxonStore" / "ghost"
        user_dir.mkdir(parents=True)
        # No store_meta.json at all
        assert get_or_create_node_id(user_dir) == ""

    def test_version_marker_no_longer_writes_hostname(self, tmp_path):
        """``bump`` must populate ``owner_node_id`` and leave
        ``owner_host`` empty — no hostname leak through the synced
        volume."""
        from axon.version_marker import bump, read

        marker = bump(tmp_path, node_id="my-uuid-1234")
        assert marker["owner_host"] == ""
        assert marker["owner_node_id"] == "my-uuid-1234"
        # And it round-trips on disk
        on_disk = read(tmp_path)
        assert on_disk is not None
        assert on_disk["owner_host"] == ""
        assert on_disk["owner_node_id"] == "my-uuid-1234"

    def test_version_marker_no_node_id_arg_writes_empty(self, tmp_path):
        from axon.version_marker import bump

        marker = bump(tmp_path)
        assert marker["owner_host"] == ""
        assert marker["owner_node_id"] == ""

    def test_version_marker_does_not_import_socket(self):
        """Defensive: a regression that re-introduces ``socket`` here
        would re-introduce the hostname-leak surface area."""
        import inspect

        import axon.version_marker as vm

        src = inspect.getsource(vm)
        assert (
            "import socket" not in src
        ), "axon.version_marker must not import socket (Item 4a leak surface)"


# ---------------------------------------------------------------------------
# 4c — random padding in AXSL sealed files
# ---------------------------------------------------------------------------


pytest.importorskip("cryptography", reason="requires axon-rag[sealed] / cryptography")


class TestSealPadding:
    def test_round_trip_with_padding(self, tmp_path):
        from axon.security.crypto import SealedFile, generate_dek

        path = tmp_path / "secret.bin"
        key = generate_dek()
        SealedFile.write(path, b"hello world", key, padding_bytes=512)
        out = SealedFile.read(path, key)
        assert out == b"hello world"

    def test_round_trip_no_padding_unchanged(self, tmp_path):
        """Default ``padding_bytes=0`` must produce the same on-disk
        layout as before Item 4c — backward compat."""
        from axon.security.crypto import SealedFile, generate_dek

        path = tmp_path / "no-pad.bin"
        key = generate_dek()
        SealedFile.write(path, b"unchanged", key)
        # File size: 16 (header) + 12 (nonce) + 9 (plaintext) + 16 (tag) = 53
        assert path.stat().st_size == 53
        assert SealedFile.read(path, key) == b"unchanged"

    def test_padding_grows_file_size(self, tmp_path):
        """A non-zero ``padding_bytes`` budget gives at least SOME
        write that grows the file beyond the no-pad baseline; over many
        writes a meaningful fraction differ in size."""
        from axon.security.crypto import SealedFile, generate_dek

        key = generate_dek()
        baseline = 16 + 12 + 5 + 16  # header + nonce + plaintext("hello") + tag
        sizes: list[int] = []
        for i in range(50):
            p = tmp_path / f"f{i}.bin"
            SealedFile.write(p, b"hello", key, padding_bytes=1024)
            sizes.append(p.stat().st_size)
        # Every file is at least the baseline; at least one is strictly bigger
        assert all(s >= baseline for s in sizes)
        assert max(sizes) > baseline
        # Distribution check: at least 20 distinct sizes (would be ~50 if
        # the RNG is healthy — but allow slack for very short bursts).
        assert len(set(sizes)) >= 20

    def test_padding_random_lengths_uniform_ish(self, tmp_path):
        """The padding length samples from a 0..N uniform distribution
        — no single length should dominate."""
        from collections import Counter

        from axon.security.crypto import SealedFile, generate_dek

        key = generate_dek()
        sizes: list[int] = []
        for i in range(200):
            p = tmp_path / f"u{i}.bin"
            SealedFile.write(p, b"x" * 8, key, padding_bytes=256)
            sizes.append(p.stat().st_size)
        most = Counter(sizes).most_common(1)[0][1]
        # No single padding length picked > 30% of the time
        assert most < len(sizes) * 0.3

    def test_streaming_write_supports_padding(self, tmp_path):
        from axon.security.crypto import SealedFile, generate_dek

        key = generate_dek()
        path = tmp_path / "stream.bin"
        chunks = [b"abc", b"def", b"ghi"]
        SealedFile.write_stream(path, iter(chunks), key, padding_bytes=128)
        assert SealedFile.read(path, key) == b"abcdefghi"

    def test_negative_padding_rejected(self, tmp_path):
        from axon.security.crypto import SealedFile, generate_dek

        with pytest.raises(ValueError, match="padding_bytes"):
            SealedFile.write(tmp_path / "x.bin", b"x", generate_dek(), padding_bytes=-1)

    def test_truncated_padding_raises_format_error(self, tmp_path):
        """If a header claims more padding than the file contains, read
        must raise ``SealedFormatError`` (not silently misalign into
        ``InvalidTag``). Build the scenario explicitly: write with
        padding_bytes=0 (file has just header+nonce+ct+tag), then
        re-stamp the header to claim padding_length=64. There are no
        trailing pad bytes in the file, so read() will fail the
        length-check before AESGCM and raise SealedFormatError."""
        from axon.security.crypto import (
            _HEADER_STRUCT,
            HEADER_LEN,
            SealedFile,
            SealedFormatError,
            generate_dek,
        )

        path = tmp_path / "truncated.bin"
        key = generate_dek()
        SealedFile.write(path, b"hello", key, padding_bytes=0)

        body = path.read_bytes()
        magic, version, cipher_id, _ = _HEADER_STRUCT.unpack(body[:HEADER_LEN])
        # Lie: claim padding_length=64 even though no pad bytes follow.
        new_header = _HEADER_STRUCT.pack(magic, version, cipher_id, 64)
        path.write_bytes(new_header + body[HEADER_LEN:])

        with pytest.raises(SealedFormatError, match="padding"):
            SealedFile.read(path, key)


# ---------------------------------------------------------------------------
# Config field for 4c
# ---------------------------------------------------------------------------


class TestPaddingConfig:
    def test_default_is_zero(self):
        from axon.config import AxonConfig

        cfg = AxonConfig()
        assert cfg.seal_padding_bytes == 0

    def test_yaml_round_trip(self, tmp_path):
        from axon.config import AxonConfig

        path = tmp_path / "config.yaml"
        path.write_text("security:\n  seal_padding_bytes: 1024\n", encoding="utf-8")
        cfg = AxonConfig.load(str(path))
        assert cfg.seal_padding_bytes == 1024
        cfg.seal_padding_bytes = 0
        cfg.save(str(path))
        cfg2 = AxonConfig.load(str(path))
        assert cfg2.seal_padding_bytes == 0

    def test_yaml_negative_rejected(self, tmp_path):
        from axon.config import AxonConfig

        path = tmp_path / "config.yaml"
        path.write_text("security:\n  seal_padding_bytes: -5\n", encoding="utf-8")
        with pytest.raises(ValueError, match="seal_padding_bytes"):
            AxonConfig.load(str(path))

    def test_yaml_above_reader_cap_rejected(self, tmp_path):
        """Copilot finding on PR #109: a budget above the reader's
        1 MiB sanity bound would emit files this build can't decrypt.
        Reject at config-load instead of letting it become silent
        data loss."""
        from axon.config import AxonConfig

        path = tmp_path / "config.yaml"
        path.write_text(f"security:\n  seal_padding_bytes: {2 * 1024 * 1024}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="1 MiB"):
            AxonConfig.load(str(path))
