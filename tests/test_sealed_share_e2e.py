"""E2E integration test for the full sealed-share roundtrip.

Exercises the complete chain in a single test:
  owner seals project → generates share string →
  grantee redeems → grantee switches to mount → grantee queries → results returned.

This is the P0 missing test before v0.3.0 release.  All crypto and
file I/O is real; only LLM calls, embedding calls, and the OS keyring
are replaced with lightweight test doubles.

Run with:
    python -m pytest tests/test_sealed_share_e2e.py -v --no-cov -s
"""
from __future__ import annotations

import getpass
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("cryptography")
pytest.importorskip("keyring")

# ---------------------------------------------------------------------------
# In-memory keyring — same pattern as test_sealed_share.py / test_project_seal.py
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def kr_backend():
    """Shared in-memory keyring used by both owner and grantee operations."""
    backend = _InMemoryKeyring()
    with patch("axon.security.keyring._keyring.get_keyring", return_value=backend):
        from axon.security import master as _master_mod

        _master_mod._unlocked_masters.clear()
        yield backend
        _master_mod._unlocked_masters.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 384  # embedding dimension used throughout


def _deterministic_embed(texts):
    """Return fixed-length vectors — all docs look the same to the ANN index
    so every result is equally "close" to the query, which is fine for
    verifying the retrieval pipeline runs end-to-end."""
    return [[0.1] * _DIM for _ in texts]


def _canned_llm_response(*_args, **_kwargs):
    return "The knowledge base contains information about quantum entanglement."


def _make_brain_config(axon_store_base: Path, vs_path: Path, bm25_path: Path):
    """Return a minimal AxonConfig whose projects_root is derived from
    *axon_store_base* (the only way to control the path without fighting
    __post_init__'s unconditional override of ``projects_root``)."""
    from axon.config import AxonConfig

    return AxonConfig(
        axon_store_base=str(axon_store_base),
        vector_store_path=str(vs_path),
        bm25_path=str(bm25_path),
        vector_store="turboquantdb",
        # Disable expensive optional strategies to keep the test fast.
        raptor=False,
        graph_rag=False,
        hybrid_search=False,
        rerank=False,
        hyde=False,
        multi_query=False,
        query_router="off",
        # Disable query cache so every query hits the retrieval pipeline.
        query_cache=False,
    )


def _make_brain(cfg):
    """Construct a real AxonBrain with mocked LLM + embedding."""
    from axon.main import AxonBrain

    with (
        patch("axon.main.OpenEmbedding") as MockEmb,
        patch("axon.main.OpenLLM") as MockLLM,
        patch("axon.main.OpenReranker"),
    ):
        mock_emb = MockEmb.return_value
        mock_emb.embed.side_effect = _deterministic_embed
        mock_emb.embed_query.return_value = [0.1] * _DIM
        mock_emb.dim = _DIM

        mock_llm = MockLLM.return_value
        mock_llm.generate.side_effect = _canned_llm_response
        mock_llm.complete.side_effect = _canned_llm_response

        brain = AxonBrain(cfg)
        # Patch the live attributes so query() also uses the mocks.
        brain.embedding = mock_emb
        brain.llm = mock_llm
        return brain


def _ingest_research_docs(brain) -> None:
    """Ingest three short documents into the active project."""
    docs = [
        {
            "id": "doc1",
            "text": "Quantum entanglement is a phenomenon where particles become correlated.",
            "metadata": {"source": "physics.txt"},
        },
        {
            "id": "doc2",
            "text": "The double-slit experiment demonstrates wave-particle duality.",
            "metadata": {"source": "physics.txt"},
        },
        {
            "id": "doc3",
            "text": "Schrödinger's cat is a thought experiment about quantum superposition.",
            "metadata": {"source": "physics.txt"},
        },
    ]
    brain.ingest(docs)


# ---------------------------------------------------------------------------
# The integration test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_sealed_share_e2e_roundtrip(kr_backend, tmp_path):
    """Full sealed-share E2E roundtrip.

    Uses a single ``tmp_path`` so both owner and grantee stores sit on
    the same filesystem (simulates OneDrive / shared drive sync without
    needing a real network).

    The OS username is used as the owner name because AxonConfig always
    derives ``projects_root`` from ``{axon_store_base}/AxonStore/{os_username}``.
    The grantee uses a separate ``axon_store_base`` directory but shares
    access to the owner's project directory via the filesystem.

    Phase 1 — Owner setup, ingest, seal
    Phase 2 — Share generation
    Phase 3 — Grantee redeem
    Phase 4 — Grantee query via sealed mount
    """
    from axon.security.master import bootstrap_store, unlock_store
    from axon.security.seal import is_project_sealed
    from axon.security.share import generate_sealed_share, redeem_sealed_share

    _username = getpass.getuser()

    # Owner's store base: tmp_path/owner_store → projects_root = tmp_path/owner_store/AxonStore/<username>
    owner_store_base = tmp_path / "owner_store"
    owner_store_base.mkdir()
    owner_user_dir = owner_store_base / "AxonStore" / _username
    owner_user_dir.mkdir(parents=True)

    # Grantee's store base: tmp_path/grantee_store → projects_root = tmp_path/grantee_store/AxonStore/<username>
    grantee_store_base = tmp_path / "grantee_store"
    grantee_store_base.mkdir()
    grantee_user_dir = grantee_store_base / "AxonStore" / _username
    grantee_user_dir.mkdir(parents=True)

    # ------------------------------------------------------------------
    # Phase 1a: bootstrap owner's store (derives master key)
    # ------------------------------------------------------------------
    bootstrap_store(owner_user_dir, "owner-secret-pw")
    # bootstrap_store leaves the store unlocked on some paths but we
    # explicitly unlock to guarantee process-local master cache is warm.
    unlock_store(owner_user_dir, "owner-secret-pw")

    # ------------------------------------------------------------------
    # Phase 1b: build owner AxonBrain, create + ingest "research" project
    # ------------------------------------------------------------------
    owner_default_vs = owner_user_dir / "_default_vs"
    owner_default_bm25 = owner_user_dir / "_default_bm25"
    owner_default_vs.mkdir()
    owner_default_bm25.mkdir()

    owner_cfg = _make_brain_config(
        axon_store_base=owner_store_base,
        vs_path=owner_default_vs,
        bm25_path=owner_default_bm25,
    )
    # Sanity check: AxonConfig derivation lands on the right user dir.
    assert (
        Path(owner_cfg.projects_root) == owner_user_dir
    ), f"projects_root mismatch: {owner_cfg.projects_root} != {owner_user_dir}"

    owner_brain = _make_brain(owner_cfg)
    try:
        # Create the "research" project under the owner's store.
        from axon.projects import ensure_project

        ensure_project("research")
        owner_brain.switch_project("research")
        _ingest_research_docs(owner_brain)
    finally:
        owner_brain.close()

    # ------------------------------------------------------------------
    # Phase 1c: seal the research project (real AES-GCM encryption)
    # ------------------------------------------------------------------
    from axon.projects import project_dir
    from axon.security.seal import project_seal

    research_dir = project_dir("research")
    seal_result = project_seal("research", owner_user_dir)
    assert seal_result["status"] == "sealed", f"Expected sealed, got: {seal_result['status']}"
    assert is_project_sealed(research_dir), "Project directory must carry the sealed marker"

    # ------------------------------------------------------------------
    # Phase 2: owner generates a sealed share
    # ------------------------------------------------------------------
    share_result = generate_sealed_share(
        owner_user_dir,
        "research",
        grantee=_username,
        key_id="ssk_e2e_test1",
    )
    share_string = share_result["share_string"]
    assert share_result["sealed"] is True
    assert share_result["key_id"] == "ssk_e2e_test1"
    assert share_result["owner"] == _username
    assert share_result["project"] == "research"

    # ------------------------------------------------------------------
    # Phase 3: grantee redeems share (no master bootstrap needed)
    # ------------------------------------------------------------------
    # redeem_sealed_share only needs the OS keyring to stash the DEK —
    # it does NOT require a bootstrapped master store on the grantee side.

    redeem_result = redeem_sealed_share(grantee_user_dir, share_string)
    assert redeem_result["sealed"] is True
    assert redeem_result["key_id"] == "ssk_e2e_test1"
    assert redeem_result["owner"] == _username
    assert redeem_result["project"] == "research"
    mount_name = redeem_result["mount_name"]
    assert mount_name, "redeem_sealed_share must return a mount_name"

    # Verify the mount descriptor was written and validates correctly.
    from axon.mounts import list_mount_descriptors, load_mount_descriptor, validate_mount_descriptor

    desc = load_mount_descriptor(grantee_user_dir, mount_name)
    assert desc is not None, "mount.json must exist after redeem"
    valid, reason = validate_mount_descriptor(desc)
    assert valid, f"Sealed mount descriptor failed canonical validation: {reason}"
    assert desc["mount_type"] == "sealed"
    assert desc["share_key_id"] == "ssk_e2e_test1"
    assert desc["readonly"] is True
    assert desc["state"] == "active"
    assert desc["revoked"] is False

    mounts_listed = list_mount_descriptors(grantee_user_dir)
    assert any(
        d.get("mount_name") == mount_name for d in mounts_listed
    ), f"Mount '{mount_name}' must appear in list_mount_descriptors"

    # ------------------------------------------------------------------
    # Phase 4: grantee brain switches to sealed mount and runs a query
    # ------------------------------------------------------------------
    grantee_default_vs = grantee_user_dir / "_default_vs"
    grantee_default_bm25 = grantee_user_dir / "_default_bm25"
    grantee_default_vs.mkdir()
    grantee_default_bm25.mkdir()

    grantee_cfg = _make_brain_config(
        axon_store_base=grantee_store_base,
        vs_path=grantee_default_vs,
        bm25_path=grantee_default_bm25,
    )
    assert (
        Path(grantee_cfg.projects_root) == grantee_user_dir
    ), f"grantee projects_root mismatch: {grantee_cfg.projects_root} != {grantee_user_dir}"

    grantee_brain = _make_brain(grantee_cfg)
    try:
        # switch_project("mounts/<mount_name>") triggers the two-phase flow:
        #   1. load_mount_descriptor → finds mount_type="sealed"
        #   2. stashes _pending_seal_mount on the brain
        #   3. close() → wipes prior _sealed_cache
        #   4. _mount_sealed_project() → get_grantee_dek() → materialize_for_read()
        #   5. config paths updated to the ephemeral plaintext cache dir
        #   6. OpenVectorStore + BM25Retriever opened from decrypted cache
        grantee_brain.switch_project(f"mounts/{mount_name}")

        # The sealed cache slot must be populated after switch.
        assert (
            grantee_brain._sealed_cache is not None
        ), "grantee brain must hold a sealed cache after switch_project on a sealed mount"

        # Run a query against the decrypted content.
        # query() calls _execute_retrieval + LLM generate; the LLM mock
        # returns the canned answer string so we verify the pipeline
        # completes without error and returns something non-empty.
        answer = grantee_brain.query("What is quantum entanglement?")
        assert isinstance(answer, str) and answer, "query() must return a non-empty string answer"
    finally:
        grantee_brain.close()

    # After close() the sealed cache must be wiped.
    assert (
        grantee_brain._sealed_cache is None
    ), "close() must set _sealed_cache to None after releasing the sealed mount"
