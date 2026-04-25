# SHARE_MOUNT_SEALED — plan for encrypted-at-rest share mounts

> **Status:** Plan / design proposal. **No implementation yet.** This doc
> exists to drive a decision on the open questions in §5 before any
> code is written. Supersedes the rejected `SHARE_MOUNT_REMOTE.md`
> (server-mediated mounts) — that approach required the owner to run a
> long-lived `axon-api` reachable by grantees, which broke the
> offline-owner workflow.
>
> Builds on the four shipped fixes from `fix/share-mount-sqlite-wal-safety`
> (issues #51–#54): cloud-sync path classifier, WAL→DELETE journal
> mode, owner-side DB relocation, version marker + active refresh,
> share-key TTL + extend. Those fixes harden the **plaintext**
> share-mount path; this plan replaces it with an **encrypted** path
> where revocation actually invalidates cached bytes.

## 1. Context

Today every byte on disk is plaintext on every OS:

```
~/.axon/AxonStore/<owner>/<project>/
  meta.json                              ← plaintext JSON
  bm25_index/.bm25_log.jsonl             ← plaintext JSONL
  bm25_index/.dynamic_graph.snapshot.json ← plaintext JSON (#51)
  vector_store_data/manifest.json        ← plaintext JSON
  vector_store_data/seg-*.bin            ← plaintext binary
  version.json                           ← plaintext JSON (#53)
```

A grantee who has read access to those bytes (via OneDrive sync, a
shared SMB folder, `cp -r`, a USB stick) can open them with any tool
that understands the format. The HMAC share token Axon issues today
is a **one-time bearer credential** for the redeem call — it does not
encrypt anything and is not presented after redemption. Revocation is
enforced only by the Axon client checking the owner's manifest; a
grantee with cached bytes and a hacked client bypasses it entirely.

`src/axon/security.py` already sketches the right interface for an
encrypted "sealed" mode (`bootstrap_store`, `unlock_store`,
`generate_sealed_share`, `redeem_sealed_share`, `project_seal`,
`project_rotate_keys`), but every function is a stub that raises
`SecurityError("not configured")`. The skeleton is there; the
implementation is not.

## 2. Hard requirements

From the design conversation:

1. **Lock the dataset with the key.** The key must do load-bearing
   cryptographic work, not just gate an Axon-client check.
2. **No API server.** Owner must not need a long-lived process for
   grantees to query. Owner offline → grantee still queries.
3. **Linux + Windows parity.** Same model on both. Same code path.
4. **Works through online drives** (OneDrive primarily; Dropbox /
   Google Drive / SMB / SSHFS as long as bytes can land on the
   grantee's disk).
5. **Revocation actually invalidates cached bytes.** Today's
   manifest-marking model is insufficient — see §1.

## 3. Threat model

| Threat | In scope (must defend) | Out of scope (v1) |
|---|---|---|
| Cached-bytes-after-revocation: grantee with synced files + revoked share + hacked client | ✅ | |
| Filesystem-level snooping: another OneDrive collaborator added for backup who is NOT a grantee | ✅ | |
| Cloud provider snooping: bytes-at-rest in OneDrive/Dropbox datacenter | ✅ | |
| Compromised owner machine (master key extracted) | | All keys leak — accept |
| Compromised grantee machine during active session (DEK in memory) | | Memory dump = data leak — accept |
| Side-channel on grantee (cache timing on encrypted reads) | | Not defending |
| Man-in-the-middle on share-string delivery (Slack/email) | | Out-of-band channel is the user's problem |
| Quantum-attacker-in-2040 | | AES-256-GCM is the line; PQC out of scope |

## 4. Architecture

### 4.1 Encryption model — AES-GCM per-file with envelope keys

**Per-file format** (every file in the project becomes a sealed file):

```
+----------+------------------+--------------+-----------+
| header   | 12-byte nonce    | ciphertext   | 16-byte   |
| (16 B)   | (random per file)| (variable)   | auth tag  |
+----------+------------------+--------------+-----------+
```

`header`: 4-byte magic `AXSL` + 1-byte version + 1-byte cipher_id (0
= AES-256-GCM) + 10 bytes reserved/zero. Header is authenticated as
AAD so a downgrade attempt fails.

**Envelope keys** (standard KMS pattern):

```
master key (256 bits, owner-only, in OS keyring)
    |
    +-- wraps --> per-project DEK (Data Encryption Key, 256 bits)
                      |
                      +-- encrypts --> every file in vector_store_data/, bm25_index/,
                                       meta.json, .dynamic_graph.snapshot.json, etc.

per-share KEK (Key Encryption Key, derived from share token via HKDF)
    |
    +-- wraps --> a copy of the project DEK
                      |
                      +-- delivered to grantee via share_string
                                       |
                                       +-- grantee unwraps using share token
                                                          |
                                                          +-- now has DEK; can decrypt files
```

Wrapping = AES-256-KW (RFC 5649) — small (40 bytes per wrap), constant time, well-trusted.

### 4.2 Key storage

| Key | Owner storage | Grantee storage |
|---|---|---|
| **Master** | OS keyring (DPAPI on Windows, Keychain on macOS, Secret Service on Linux) with passphrase fallback at `~/.axon/.security/master.enc` | (Grantees never see the master key) |
| **Project DEK (encrypted with master)** | `~/.axon/AxonStore/<owner>/<project>/.security/dek.wrapped` | (Not stored — derived from share material at mount time) |
| **Project DEK (encrypted with per-share KEK)** | `~/.axon/AxonStore/<owner>/<project>/.security/shares/<key_id>.wrapped` | Bytes synced to grantee via OneDrive; same path on grantee side |
| **Share token (used to derive KEK)** | (Owner generates, transmits via share_string) | OS keyring at `axon.share.<key_id>` after redemption |
| **Plaintext DEK in process memory** | During open project | During mount session (cleared on switch / close) |

The owner's master key never leaves the owner's machine. The
per-project DEK is stored on the shared filesystem **only in
encrypted form** — wrapped by the master (for the owner) and by each
share's KEK (for each grantee).

### 4.3 Sealed-share lifecycle

**Owner: seal an existing project (one-time migration)**
```
axon project seal research
  → derives a fresh DEK
  → wraps DEK with master, writes .security/dek.wrapped
  → encrypts every file in research/ in place (atomic per-file: write to .sealing, fsync, rename)
  → marks project as sealed in meta.json (cipher_suite=AES-256-GCM, schema=v1)
```

**Owner: generate a sealed share**
```
axon share generate research bob --ttl-days 30 --sealed
  → derives a per-share token (32 random bytes, like today)
  → derives KEK = HKDF(token, salt=key_id, info="axon-share-v1")
  → wraps DEK with KEK, writes .security/shares/<key_id>.wrapped
  → builds share_string carrying (key_id, token, owner, project, owner_store_path, sealed=true)
  → records share in .share_keys.json + .share_manifest.json (TTL, revocation)
```

**Grantee: redeem**
```
axon share redeem <share_string>
  → parses share_string, extracts token + key_id
  → reads .security/shares/<key_id>.wrapped from owner's synced folder
  → derives KEK from token + key_id, unwraps DEK
  → stores DEK in grantee's OS keyring at axon.share.<key_id>
  → token is wiped from process memory — only the unwrapped DEK is kept (in keyring)
  → writes mount.json with mount_type="sealed" (no plaintext key on disk)
```

**Grantee: query a sealed mount**
```
axon /project switch mounts/alice_research
  → reads mount.json, sees mount_type="sealed"
  → fetches DEK from OS keyring (axon.share.<key_id>)
  → opens vector_store / bm25 via SealedFileReader, which decrypts each file on read
```

**Owner: revoke (soft) — fast, doesn't invalidate cached bytes**
```
axon share revoke sk_a1b2c3d4
  → marks revoked=true in .share_manifest.json
  → optionally deletes .security/shares/sk_a1b2c3d4.wrapped (so a fresh redeem fails)
  → grantee with cached DEK in keyring CAN still decrypt cached files
  → DEK rotation needed for hard revocation (see below)
```

**Owner: revoke (hard) — slow, invalidates cached bytes**
```
axon share revoke sk_a1b2c3d4 --rotate
  → derives a new project DEK (new_DEK)
  → re-encrypts every file in the project with new_DEK (atomic per-file)
  → wraps new_DEK with master + with KEK of every still-valid share
  → DELETES the old DEK wrappers (including the revoked grantee's)
  → revoked grantee's cached DEK is now useless: all on-disk files use new_DEK
```

### 4.4 Mmap policy — decrypt-into-memory at open

mmap on encrypted bytes is impossible without a transparent decryption
layer (FUSE on Linux, WinFsp/Dokany on Windows — operationally complex,
extra dependency, breaks the "no special install" property).

**v1 policy: decrypt the whole file into RAM at `open()`.** Costs:
- 1 GB vector store on a 16 GB grantee: fine.
- 10 GB on an 8 GB grantee: out of memory. Document the limit.

**v2 (deferred):** ephemeral plaintext cache in `tempfile.mkdtemp()` /
`%TEMP%`, mmap the cache, wipe on close. Re-introduces a session-bounded
plaintext footprint, but bounded in time.

### 4.5 Backend compatibility

| Backend | v1 plan | Why |
|---|---|---|
| **TurboQuantDB** (default) | Supported | Manifest + segment files + WAL log all wrappable. Decrypt-into-memory at `Database.open()`. |
| **LanceDB** | Phase 3 (after TQDB stable) | Append-only fragments make encryption clean. Compaction needs to re-encrypt. |
| **Chroma** (SQLite) | NOT supported in sealed mode (v1) | SQLite needs random in-place writes; whole-file encryption defeats it. SQLCipher is the right answer but is a separate, bigger effort. Preflight (#52) extended to also reject sealed+Chroma. |
| **BM25** (msgpack/JSONL) | Supported | Small files, naive read-all-decrypt-all is fine. |
| **DynamicGraphBackend** | Snapshot file encrypted; owner DB stays local-only (already done in #51) | The grantee only ever reads the snapshot, which is a small JSON. |
| **Governance audit DB** | Use the existing JSONL fallback (#51); JSONL lines are individually encrypted so partial-sync corruption is bounded | SQLite-WAL on encrypted bytes is double-bad |

### 4.6 What sits where on the shared filesystem

After sealing, the project layout becomes:

```
~/.axon/AxonStore/<owner>/<project>/
  meta.json                              ← AXSL header + ciphertext
  .security/
    dek.wrapped                          ← 40 bytes, owner-only (owner has master)
    shares/
      sk_a1b2c3d4.wrapped                ← 40 bytes, per-share DEK wrap
      sk_eeff0011.wrapped                ← (one per active share)
  bm25_index/
    .bm25_log.jsonl                      ← AXSL-wrapped
    .dynamic_graph.snapshot.json         ← AXSL-wrapped
  vector_store_data/
    manifest.json                        ← AXSL-wrapped
    seg-*.bin                            ← AXSL-wrapped
    live_*.bin                           ← AXSL-wrapped (decrypted into RAM at open)
  version.json                           ← still PLAINTEXT (so grantees can detect changes without DEK)
```

`version.json` stays plaintext deliberately so the existing
`MountSyncPendingError` machinery from #53 keeps working — grantees
need to be able to detect "owner has re-ingested" before they bother
fetching the DEK and decrypting. The marker contains only hashes of
the (encrypted) files, which leaks change-frequency but not content.

## 5. Decisions needed before code

I have a recommendation for each. Confirm or override.

### 5.1 Mmap policy
- Option A: **Decrypt-into-memory at open (RAM-bounded).** Simple, no temp files, works the same on Linux/Windows.
- Option B: Decrypt to ephemeral plaintext cache, mmap the cache.

**Recommendation:** A for v1. Add B as an optimization if users hit RAM limits on >5GB projects.

### 5.2 Crypto library
- Option A: **`cryptography` PyPI package** (~1 MB, well-audited, AES-256-GCM + AES-KW native). Already a transitive dep of `langchain` and several other things in our extras.
- Option B: stdlib only (no AES-GCM directly — would need to build on `secrets` + a pure-Python AES; slow + risky).
- Option C: Rust extension via the existing `axon_rust` module.

**Recommendation:** A. Mature, audited, fast enough. Rust route only if benchmarking shows `cryptography` is the bottleneck.

### 5.3 Key storage on the grantee side
- Option A: **OS keyring** (`keyring` PyPI package — DPAPI on Windows, Keychain on macOS, Secret Service on Linux). User-friendly; key follows the user's OS account.
- Option B: Passphrase-protected file (`~/.axon/.security/grantee.enc`). User types passphrase at switch time.
- Option C: Both — keyring with passphrase fallback when keyring is unavailable.

**Recommendation:** C. Keyring as default, passphrase as graceful fallback (e.g. headless servers without Secret Service).

### 5.4 Backend scope for v1
- Option A: **TurboQuantDB only.** Document Chroma+sealed as unsupported (extend preflight). LanceDB in Phase 3.
- Option B: TQDB + LanceDB simultaneously.
- Option C: All three including Chroma via SQLCipher.

**Recommendation:** A. TQDB is the default backend; covers most users. LanceDB is one more phase. Chroma requires SQLCipher work that should be its own decision.

### 5.5 Revocation cost — soft vs hard
- Both available, with clear UX:
  - `axon share revoke sk_xxx` (soft) — fast; manifest mark only; does NOT invalidate cached bytes; explicit warning in the output
  - `axon share revoke sk_xxx --rotate` (hard) — slow; rotates DEK + re-encrypts; cached bytes become useless

**Recommendation:** Ship both, default to soft, require `--rotate` to be explicit. The doc + REPL output spell out exactly what each means.

### 5.6 Migration of existing plaintext projects
- Option A: `axon project seal <name>` — encrypts in place. Atomic per-file (write to `.sealing` sibling, fsync, rename). On crash, recovery scans for `.sealing` files and resumes.
- Option B: Force a fresh re-ingest into a sealed project (no migration).

**Recommendation:** A. Re-ingest is too painful for users with hours of LLM work invested.

## 6. Phased implementation

Each phase = one focused PR. Phases 1–2 are the MVP that delivers
end-to-end encryption for a single backend (TQDB) on a single owner;
later phases add reach and polish.

| Phase | Deliverable | Why this order |
|---|---|---|
| **0** | This doc + decisions in §5 finalised + opens GitHub issue per phase | Don't start coding until §5 is locked |
| **1** | Crypto foundations: `axon.security.crypto` module (`SealedFile.{open,write}`, AES-KW wrap/unwrap, KEK derivation), master keyring integration, tests on synthetic data. **Zero behavior change for existing projects.** | Lays the cryptographic primitives all later phases use |
| **2** | TurboQuantDB sealed read/write path — `axon project seal` works for a single TQDB project; owner can open/query their own sealed project; tests | First end-to-end flow; proves the design |
| **3** | Sealed-share generation + redemption + mount flow — wire `generate_sealed_share` / `redeem_sealed_share` (currently stubs); grantee can query a sealed project on a shared filesystem | Closes the user-visible loop; manual two-machine smoke runnable |
| **4** | Soft + hard revocation — `axon share revoke [--rotate]`; rotation re-encrypts; cached bytes invalidated; tests | Closes the security promise |
| **5** | Cross-interface surfaces (REST `/store/seal`, `/share/generate?sealed=true`, `/share/revoke?rotate=true`; MCP tools; REPL `/store seal`, `/share generate --sealed`, `/share revoke --rotate`; CLI flags); docs (`SHARE_MOUNT.md` rewrite) | Cross-interface parity rule |
| **6** | LanceDB sealed support; preflight extends to reject sealed+Chroma loudly | Second backend |
| **7** | Verification: extend `SHARE_MOUNT_SMOKE.md` with sealed-store steps; real two-machine OneDrive run before tagging | Empirical proof on the actual target environment |

Each phase has its own GitHub issue (templated after #51–#54).
Total estimated work: ~3–5 weeks of focused effort across the 7
phases. Phases 1, 2, 3 are blocking; 4–7 can be parallelised once the
foundation is in.

## 7. Verification plan

The CI test suite covers:
- Crypto unit tests (round-trip, tamper detection, wrong-key fail, header-version-downgrade rejection)
- Key-management tests with a mocked `keyring` backend (no real OS calls)
- Backend-level encrypted-file open/read tests for TQDB
- Synthetic two-machine flow (two tmpdirs, owner seals + shares, grantee redeems + queries)
- Revocation tests (soft = grantee with cached DEK still reads; hard = grantee gets `cryptography.exceptions.InvalidTag`)

What CI cannot cover:
- Real OneDrive mid-sync behavior
- Real OS keyring on every platform
- Real two-machine clock skew

Those go in **`SHARE_MOUNT_SMOKE.md` v2** — extended with sealed-store
steps, run before any release that touches `axon.security`. The
existing smoke recipe (Path A) becomes obsolete once sealed mode is
the default; until then both recipes coexist.

## 8. Trade-offs summary

| Concern | Plaintext today | Sealed (this plan) | Server-mediated (rejected) |
|---|---|---|---|
| Owner must run a server | No | No | **Yes** (rejected) |
| Grantee can query offline | Yes (with cached files) | Yes (with cached files + cached DEK) | No |
| Revocation invalidates cached bytes | No | Yes (with `--rotate`) | Yes (immediate) |
| Cloud provider can read bytes | Yes (plaintext on OneDrive) | No (AES-GCM ciphertext) | Yes (only owner has the data) |
| Mmap works on grantee | Yes | No (decrypt-into-memory) | N/A (no local files) |
| Memory cost on grantee | Mmap, low | RAM = file size | Low (HTTP responses) |
| Cross-platform (Linux + Windows) | Yes (current) | Yes | Yes |
| Implementation cost | (already shipped) | **3–5 weeks** | 2–3 weeks |
| Breaking change for existing projects | n/a | Opt-in (`project seal`); plaintext keeps working | Opt-in |

## 9. Out of scope (explicit non-goals for v1)

- **SQLCipher / Chroma encryption.** Separate, bigger project. Sealed
  mode requires switching to TQDB or LanceDB until then.
- **FUSE / WinFsp transparent decryption.** v2 optimization.
- **Hardware key support** (YubiKey, TPM-bound keys). Owner keyring is
  fine for v1.
- **Post-quantum crypto.** AES-256-GCM is the line; PQC is its own
  decade.
- **Streaming encryption** (decrypt as bytes are read from disk
  block-by-block). Decrypt-into-memory is simpler and v1 has the RAM
  budget.
- **Multi-owner / write access for grantees.** Mounts stay read-only
  by design.
- **Key escrow / recovery** (e.g. Shamir-secret-sharing the master
  key across N admins). Lose master = lose project. Document, don't
  fix.

## 10. Pre-implementation checklist

- [ ] §5 decisions confirmed (5.1 through 5.6) by user
- [ ] Open GitHub issues for Phases 1–7 with this plan linked
- [ ] Add `cryptography` and `keyring` to `pyproject.toml` `[project.optional-dependencies]` under a new `sealed` extra
- [ ] Settle naming: "sealed share" vs "encrypted share" — current code uses "sealed", this plan follows that
- [ ] Confirm `src/axon/security.py` is the canonical home (rename or move if not)
- [ ] Decide on AAD content for AES-GCM (recommend: `key_id || file_relpath` so files can't be swapped between projects)
