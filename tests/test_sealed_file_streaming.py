"""Tests for SealedFile.write_stream and write_stream_from_path.

Covers the streaming encryption path added to :mod:`axon.security.crypto`:

- Round-trip: write_stream → read() → original plaintext
- write_stream_from_path → read() → original plaintext
- Various chunk sizes (1 byte, 1 KiB, 1 MiB)
- Empty plaintext (zero-byte iterator)
- Multi-chunk payload (4 MiB, split over several 1 MiB chunks)
- AAD is enforced exactly as in the non-streaming write()
- Atomic write: no .sealing tempfile left after success
- Tampered ciphertext produced by write_stream raises InvalidTag on read
- Invalid key / chunk_size arguments are rejected eagerly

Skips the entire module when the optional ``cryptography`` package
isn't installed (the ``sealed`` extra hasn't been pulled in).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from axon.security.crypto import (  # noqa: E402
    HEADER_LEN,
    NONCE_LEN,
    STREAMING_CHUNK_SIZE,
    TAG_LEN,
    SealedFile,
    generate_dek,
    make_aad,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_payload(n_bytes: int) -> bytes:
    """Return *n_bytes* of OS-random data."""
    return os.urandom(n_bytes)


def _iter_chunks(data: bytes, chunk_size: int):
    """Yield *data* in chunks of *chunk_size* bytes."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------


class TestWriteStreamRoundTrip:
    """write_stream → read() produces the original plaintext."""

    def test_basic_round_trip(self, tmp_path):
        key = generate_dek()
        payload = b"hello streaming sealed axon"
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([payload]), key)
        assert SealedFile.read(path, key) == payload

    def test_round_trip_with_aad(self, tmp_path):
        key = generate_dek()
        payload = b"authenticated additional data test"
        aad = make_aad("sk_abc", "vector_store/manifest.json")
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([payload]), key, aad=aad)
        assert SealedFile.read(path, key, aad=aad) == payload

    def test_creates_parent_dirs(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "nested" / "deep" / "out.bin"
        SealedFile.write_stream(path, iter([b"x"]), key)
        assert path.is_file()

    def test_overwrites_existing_file(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"first"]), key)
        SealedFile.write_stream(path, iter([b"second"]), key)
        assert SealedFile.read(path, key) == b"second"

    def test_str_path_accepted(self, tmp_path):
        key = generate_dek()
        path = str(tmp_path / "out.bin")
        SealedFile.write_stream(path, iter([b"str path ok"]), key)
        assert SealedFile.read(path, key) == b"str path ok"


# ---------------------------------------------------------------------------
# Empty plaintext
# ---------------------------------------------------------------------------


class TestWriteStreamEmptyPayload:
    """An empty iterator is valid — the file still has the AXSL envelope."""

    def test_empty_iterator_round_trips_to_empty_bytes(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "empty.bin"
        SealedFile.write_stream(path, iter([]), key)
        assert SealedFile.read(path, key) == b""

    def test_empty_file_has_minimum_size(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "empty.bin"
        SealedFile.write_stream(path, iter([]), key)
        # Header + nonce + (0-byte ciphertext) + tag
        assert path.stat().st_size == HEADER_LEN + NONCE_LEN + TAG_LEN

    def test_single_empty_chunk_is_treated_as_empty(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "empty_chunk.bin"
        SealedFile.write_stream(path, iter([b""]), key)
        assert SealedFile.read(path, key) == b""


# ---------------------------------------------------------------------------
# Chunk-size variants
# ---------------------------------------------------------------------------


class TestWriteStreamChunkSizes:
    """Correctness must hold regardless of how the plaintext is chunked."""

    PAYLOAD = _random_payload(16 * 1024)  # 16 KiB — fast, multi-chunk even at 1 KiB

    @pytest.mark.parametrize(
        "chunk_size",
        [1, 7, 512, 1024, 4096, STREAMING_CHUNK_SIZE],
        ids=["1B", "7B", "512B", "1KiB", "4KiB", "default"],
    )
    def test_round_trip_at_chunk_size(self, tmp_path, chunk_size):
        key = generate_dek()
        path = tmp_path / f"cs_{chunk_size}.bin"
        SealedFile.write_stream(path, _iter_chunks(self.PAYLOAD, chunk_size), key)
        assert SealedFile.read(path, key) == self.PAYLOAD


# ---------------------------------------------------------------------------
# Multi-chunk / large payload
# ---------------------------------------------------------------------------


class TestWriteStreamLargePayload:
    """Use 4 MiB to exercise multi-chunk behaviour without slowing CI."""

    def test_4mib_round_trip(self, tmp_path):
        payload = _random_payload(4 * 1024 * 1024)
        key = generate_dek()
        path = tmp_path / "big.bin"
        # Feed 1 MiB chunks so we traverse the loop multiple times.
        SealedFile.write_stream(path, _iter_chunks(payload, 1024 * 1024), key)
        assert SealedFile.read(path, key) == payload

    def test_4mib_with_aad(self, tmp_path):
        payload = _random_payload(4 * 1024 * 1024)
        key = generate_dek()
        aad = make_aad("sk_big", "large_file.bin")
        path = tmp_path / "big_aad.bin"
        SealedFile.write_stream(path, _iter_chunks(payload, 512 * 1024), key, aad=aad)
        assert SealedFile.read(path, key, aad=aad) == payload


# ---------------------------------------------------------------------------
# write_stream_from_path
# ---------------------------------------------------------------------------


class TestWriteStreamFromPath:
    """Convenience wrapper that reads src_path in chunks."""

    def test_basic_round_trip(self, tmp_path):
        payload = _random_payload(2 * 1024 * 1024)  # 2 MiB
        key = generate_dek()
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.sealed"
        src.write_bytes(payload)
        SealedFile.write_stream_from_path(src, dst, key)
        assert SealedFile.read(dst, key) == payload

    def test_with_aad(self, tmp_path):
        payload = b"sealed with aad via path"
        key = generate_dek()
        aad = make_aad("sk_path", "test/data.bin")
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.sealed"
        src.write_bytes(payload)
        SealedFile.write_stream_from_path(src, dst, key, aad=aad)
        assert SealedFile.read(dst, key, aad=aad) == payload

    def test_custom_chunk_size(self, tmp_path):
        payload = _random_payload(3 * 1024 * 1024)
        key = generate_dek()
        src = tmp_path / "src.bin"
        dst = tmp_path / "dst.sealed"
        src.write_bytes(payload)
        SealedFile.write_stream_from_path(src, dst, key, chunk_size=256 * 1024)
        assert SealedFile.read(dst, key) == payload

    def test_empty_src_file(self, tmp_path):
        key = generate_dek()
        src = tmp_path / "empty_src.bin"
        dst = tmp_path / "empty_dst.sealed"
        src.write_bytes(b"")
        SealedFile.write_stream_from_path(src, dst, key)
        assert SealedFile.read(dst, key) == b""

    def test_str_paths_accepted(self, tmp_path):
        payload = b"str path test"
        key = generate_dek()
        src = str(tmp_path / "src.bin")
        dst = str(tmp_path / "dst.sealed")
        Path(src).write_bytes(payload)
        SealedFile.write_stream_from_path(src, dst, key)
        assert SealedFile.read(dst, key) == payload

    def test_invalid_chunk_size_raises(self, tmp_path):
        key = generate_dek()
        src = tmp_path / "src.bin"
        src.write_bytes(b"x")
        with pytest.raises(ValueError, match="chunk_size"):
            SealedFile.write_stream_from_path(src, tmp_path / "dst.sealed", key, chunk_size=0)


# ---------------------------------------------------------------------------
# Atomic write behaviour
# ---------------------------------------------------------------------------


class TestWriteStreamAtomic:
    def test_no_sealing_tempfile_after_success(self, tmp_path):
        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"payload"]), key)
        names = {p.name for p in tmp_path.iterdir()}
        assert "out.bin.sealing" not in names
        assert "out.bin" in names


# ---------------------------------------------------------------------------
# AAD enforcement
# ---------------------------------------------------------------------------


class TestWriteStreamAadEnforcement:
    def test_wrong_aad_on_read_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"secret"]), key, aad=b"correct-aad")
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key, aad=b"wrong-aad")

    def test_no_aad_on_read_when_written_with_aad_raises(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"x"]), key, aad=b"aad-present")
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)  # missing AAD

    def test_no_aad_on_write_aad_on_read_raises(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"data"]), key)
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key, aad=b"unexpected-aad")


# ---------------------------------------------------------------------------
# Tamper / integrity
# ---------------------------------------------------------------------------


class TestWriteStreamIntegrity:
    def test_tampered_ciphertext_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        payload = b"integrity check payload" * 100
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, _iter_chunks(payload, 512), key)
        body = bytearray(path.read_bytes())
        # Flip a bit right after the nonce — inside the ciphertext region.
        body[HEADER_LEN + NONCE_LEN + 1] ^= 0xFF
        path.write_bytes(bytes(body))
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)

    def test_tampered_tag_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"tag integrity test"]), key)
        body = bytearray(path.read_bytes())
        body[-1] ^= 0xFF  # flip a bit in the trailing GCM tag
        path.write_bytes(bytes(body))
        with pytest.raises(InvalidTag):
            SealedFile.read(path, key)

    def test_wrong_key_raises_invalid_tag(self, tmp_path):
        from cryptography.exceptions import InvalidTag

        key = generate_dek()
        wrong_key = generate_dek()
        path = tmp_path / "out.bin"
        SealedFile.write_stream(path, iter([b"wrong key test"]), key)
        with pytest.raises(InvalidTag):
            SealedFile.read(path, wrong_key)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestWriteStreamValidation:
    def test_wrong_key_length_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="32 bytes"):
            SealedFile.write_stream(tmp_path / "x.bin", iter([b"data"]), b"short-key")

    def test_non_positive_chunk_size_raises_value_error(self, tmp_path):
        key = generate_dek()
        with pytest.raises(ValueError, match="chunk_size"):
            SealedFile.write_stream(tmp_path / "x.bin", iter([b"data"]), key, chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size"):
            SealedFile.write_stream(tmp_path / "x.bin", iter([b"data"]), key, chunk_size=-1)


# ---------------------------------------------------------------------------
# Format compatibility: write() and write_stream() produce identical layout
# ---------------------------------------------------------------------------


class TestFormatCompatibility:
    """Both writers must produce output readable by read()."""

    def test_write_then_read_via_streaming_writer(self, tmp_path):
        """A file written by write() is readable — baseline sanity check."""
        key = generate_dek()
        payload = b"compatibility payload"
        path_a = tmp_path / "a.bin"
        SealedFile.write(path_a, payload, key)
        assert SealedFile.read(path_a, key) == payload

    def test_write_stream_produces_valid_axsl_magic(self, tmp_path):
        """write_stream() files start with the AXSL magic bytes."""
        from axon.security.crypto import MAGIC

        key = generate_dek()
        path = tmp_path / "stream.bin"
        SealedFile.write_stream(path, iter([b"magic check"]), key)
        body = path.read_bytes()
        assert body[:4] == MAGIC

    def test_both_writers_readable_by_read(self, tmp_path):
        """write() and write_stream() files are both readable by read()."""
        key = generate_dek()
        payload = _random_payload(8 * 1024)  # 8 KiB

        path_bulk = tmp_path / "bulk.bin"
        path_stream = tmp_path / "stream.bin"

        SealedFile.write(path_bulk, payload, key)
        SealedFile.write_stream(path_stream, _iter_chunks(payload, 1024), key)

        assert SealedFile.read(path_bulk, key) == payload
        assert SealedFile.read(path_stream, key) == payload

    def test_file_sizes_match_for_same_payload(self, tmp_path):
        """write() and write_stream() produce files of the same size."""
        key = generate_dek()
        payload = _random_payload(4 * 1024)  # 4 KiB

        path_bulk = tmp_path / "bulk.bin"
        path_stream = tmp_path / "stream.bin"

        SealedFile.write(path_bulk, payload, key)
        SealedFile.write_stream(path_stream, _iter_chunks(payload, 512), key)

        assert path_bulk.stat().st_size == path_stream.stat().st_size
