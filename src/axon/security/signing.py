"""Ed25519 signing keys derived deterministically from the owner's master.

This module is the foundation of v0.4.0's TTL-gated sealed-share design.
The owner needs a way to sign expiry sidecars (and, in future, any other
authoritative metadata about a share) such that:

1. The grantee can verify the signature **without** trusting the sync
   path — they have the public key embedded in the share string the
   owner sent over a secure channel.
2. The signing key is **derivable**, not separately persisted — no new
   files on disk, no new failure modes around lost key material. As
   long as the owner can unlock their store (i.e. has the master key
   in memory), they can re-derive the same Ed25519 keypair.
3. The derivation is **domain-separated** from the per-share KEK
   derivation in :mod:`axon.security.share` so a compromise of the
   signing key cannot reveal share-wrapping material and vice versa.

Wire-level:

- IKM      = owner's master key (32 bytes)
- Salt     = fixed constant (``_SIGNING_SALT``) — owner-specific
             determinism comes from the master, not the salt
- Info     = ``b"axon-share-signing-v1"`` — distinct from the share
             KEK info ``b"axon-share-v1"`` so an HKDF oracle on either
             primitive can't be cross-applied
- Output   = 32 bytes used as the Ed25519 seed
- Keypair  = ``Ed25519PrivateKey.from_private_bytes(seed)``

Public-key wire format embedded in the SEALED2 share string is the
raw 32-byte Ed25519 public key, hex-encoded → 64 ASCII hex chars.
"""
from __future__ import annotations

from pathlib import Path

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
except ImportError as exc:  # pragma: no cover — import-time guard
    raise ImportError(
        "axon.security.signing requires the 'cryptography' package. "
        "Install with: pip install axon-rag[sealed]"
    ) from exc

from . import SecurityError
from .master import get_master_key

__all__ = [
    "SIGNING_HKDF_INFO",
    "SIGNING_PUBKEY_HEX_LEN",
    "derive_signing_keypair",
    "get_signing_pubkey_hex",
    "pubkey_to_hex",
    "pubkey_from_hex",
]


# Domain-separate from the per-share KEK derivation (``b"axon-share-v1"``
# in axon.security.share). Bumping the suffix forces a clean break if
# we ever need to migrate keypairs.
SIGNING_HKDF_INFO: bytes = b"axon-share-signing-v1"

# Fixed salt — determinism per owner comes from the master key itself,
# so a static salt is appropriate and intentional. Using a random salt
# would make the keypair non-derivable and force us to persist it,
# defeating the "no new files" property of the design.
_SIGNING_SALT: bytes = b"axon-signing-v1-salt"

# Ed25519 seed length matches the public-key length: 32 bytes.
_ED25519_SEED_LEN: int = 32

# Hex-encoded public key length = 32 bytes * 2 = 64 chars.
SIGNING_PUBKEY_HEX_LEN: int = 64


def derive_signing_keypair(master: bytes) -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Derive a deterministic Ed25519 keypair from *master*.

    Args:
        master: The owner's 32-byte master key (as returned by
            :func:`axon.security.master.get_master_key`).

    Returns:
        ``(private_key, public_key)``. Same *master* always produces
        the same keypair.

    Raises:
        SecurityError: *master* is not 32 bytes.
    """
    if not isinstance(master, bytes | bytearray):
        raise SecurityError(f"master must be bytes, got {type(master).__name__}")
    if len(master) != 32:
        raise SecurityError(
            f"master must be 32 bytes (got {len(master)}); " "did you pass the wrong key material?"
        )
    seed = HKDF(
        algorithm=hashes.SHA256(),
        length=_ED25519_SEED_LEN,
        salt=_SIGNING_SALT,
        info=SIGNING_HKDF_INFO,
    ).derive(bytes(master))
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    return private_key, private_key.public_key()


def pubkey_to_hex(pubkey: Ed25519PublicKey) -> str:
    """Encode an Ed25519 public key as 64-char lowercase hex.

    Used by :mod:`axon.security.share` when building SEALED2 envelopes.
    """
    raw = pubkey.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return raw.hex()


def pubkey_from_hex(hex_str: str) -> Ed25519PublicKey:
    """Decode a 64-char hex string into an Ed25519 public key.

    Used by the grantee when parsing a SEALED2 envelope: the pubkey
    field is the only authoritative source of "who can sign for this
    share". Rejects inputs of the wrong length or shape so a malformed
    SEALED2 string fails fast.

    Raises:
        SecurityError: *hex_str* is not exactly
            :data:`SIGNING_PUBKEY_HEX_LEN` lowercase hex characters,
            or the bytes don't form a valid Ed25519 public key.
    """
    if not isinstance(hex_str, str):
        raise SecurityError(f"pubkey_hex must be str, got {type(hex_str).__name__}")
    if len(hex_str) != SIGNING_PUBKEY_HEX_LEN:
        raise SecurityError(
            f"pubkey_hex must be exactly {SIGNING_PUBKEY_HEX_LEN} chars, " f"got {len(hex_str)}"
        )
    try:
        raw = bytes.fromhex(hex_str)
    except ValueError as exc:
        raise SecurityError(f"pubkey_hex is not valid hex: {exc}") from exc
    try:
        return Ed25519PublicKey.from_public_bytes(raw)
    except Exception as exc:  # cryptography raises ValueError / InternalError
        raise SecurityError(f"pubkey bytes are not a valid Ed25519 key: {exc}") from exc


def get_signing_pubkey_hex(owner_user_dir: Path) -> str:
    """Return the hex-encoded Ed25519 public key for *owner_user_dir*.

    Convenience wrapper that loads the master and derives the keypair.
    Use :func:`derive_signing_keypair` directly + :func:`pubkey_to_hex`
    if you already have the master in scope (saves a keyring round-trip).

    Args:
        owner_user_dir: Owner's AxonStore user directory.

    Returns:
        64-character lowercase hex string. Suitable to inline in a
        colon-delimited share envelope.

    Raises:
        SecurityError: store is not unlocked, or the master key is
            otherwise unavailable.
    """
    master = get_master_key(Path(owner_user_dir))
    _privkey, pubkey = derive_signing_keypair(master)
    return pubkey_to_hex(pubkey)
