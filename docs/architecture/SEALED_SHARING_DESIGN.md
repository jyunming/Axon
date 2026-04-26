# SEALED_SHARING_DESIGN — plan for encrypted-at-rest share mounts

> **Status:** Phases 1–6 shipped; Phase 7 (smoke verification) shipped.
> All cryptographic code, key-management, share lifecycle, ephemeral
> cache, cross-interface surfaces, and passphrase fallback are
> production-ready. Phase 7 adds the two-machine OneDrive/GDrive
> smoke procedure and integration tests for ``switch_project``'s
> sealed path. Supersedes the rejected ``SHARE_MOUNT_REMOTE.md``
> (server-mediated mounts) — that approach required the owner to run
> a long-lived ``axon-api`` reachable by grantees, which broke the
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
| **Master** | OS keyring (DPAPI on Windows, Keychain on macOS, Secret Service on Linux) with passphrase fallback at `~/.axon/AxonStore/<username>/.security/master.enc` | (Grantees never see the master key) |
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

### 4.4 Mmap policy — ephemeral plaintext cache (decision locked 2026-04-25)

mmap on encrypted bytes is impossible without a transparent decryption
layer (FUSE on Linux, WinFsp/Dokany on Windows — operationally complex,
extra dependency, breaks the "no special install" property).

**v1 policy: ephemeral plaintext cache in OS temp dir, mmap'd by
backends, wiped on close.** Decided over the simpler "decrypt-into-RAM
at open" alternative because:

- mmap continues to work natively → TurboQuantDB and LanceDB perf is
  unchanged from plaintext mode.
- No RAM bound — works on 10 GB+ projects on a 4 GB grantee.
- **Collapses backend integration cost**: with the cache approach the
  layer is "decrypt to cache, point backend at cache path" — works
  identically for TQDB and LanceDB. (This is what made the wider
  backend scope in §5.4 affordable.)

Cost: a session-bounded plaintext footprint on disk. Mitigations
(Phase 2 work):

- **Cache location**: `tempfile.mkdtemp(prefix="axon-sealed-")` on
  Linux/macOS; `%LOCALAPPDATA%\Temp\axon-sealed-<uuid>` on Windows.
  Per-mount; never shared between projects.
- **Wipe on close**: secure delete (overwrite with zeros, then
  unlink) so the OS doesn't leave plaintext blocks in unallocated
  space. Implemented in pure Python — no `srm`/`cipher` dep.
- **Crash recovery**: a startup scanner (run from `AxonBrain.__init__`)
  walks the temp dir for `axon-sealed-*` prefixes whose owner process
  is no longer alive (PID-based heuristic) and wipes them.
- **Capacity check**: before decrypting, verify free disk space ≥
  project size + 10% headroom; refuse with a clear error otherwise.

### 4.5 Backend compatibility (decision locked 2026-04-25: TQDB + LanceDB together)

With the ephemeral-cache decision (§4.4), backend integration is
backend-agnostic — both TQDB and LanceDB just read from the cache
directory the seal layer prepares for them. Both ship in v1.

| Backend | v1 plan | Why |
|---|---|---|
| **TurboQuantDB** (default) | Supported in v1 (Phase 2) | Backend opens the cache dir as if it were a normal project. Manifest + segment + WAL files all decrypted into the cache. |
| **LanceDB** | Supported in v1 (Phase 2) | Append-only fragments + manifest decrypted into the cache; fragments stay immutable so the cache is mostly write-once. Compaction (`optimize`) needs to re-encrypt → re-cache. |
| **Chroma** (SQLite) | NOT supported in sealed mode (v1) | SQLite needs random in-place writes; whole-file encryption + cache writeback model is a poor fit. SQLCipher integration is a separate, bigger effort. Preflight (#52) extended to also reject sealed+Chroma. |
| **BM25** (msgpack/JSONL) | Supported in v1 | Small files, fits the cache model trivially. |
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

## 5. Decisions — LOCKED 2026-04-25

The four open decisions were resolved on 2026-04-25 after Phase 1
landed. The original recommendations are kept in the table below for
context; the **Decision** column is authoritative going forward.

| § | Question | Decision | Notes |
|---|---|---|---|
| **5.1** | Mmap policy | **Ephemeral plaintext cache in OS temp dir** | NOT decrypt-into-RAM. Cache wipe on close; PID-based crash-recovery scanner; capacity check. Backends mmap the cache directly so perf is unchanged. See §4.4. |
| **5.2** | Crypto library | **`cryptography` PyPI** | Shipped Phase 1 (PR #56). |
| **5.3** | Grantee key storage | **OS keyring (Phase 1) + passphrase fallback (Phase 2)** | Phase 1 shipped. Headless-Linux fallback is Phase 2. |
| **5.4** | Backend scope v1 | **TurboQuantDB + LanceDB simultaneously** | NOT TQDB-only. Affordable because the cache decision (5.1) makes the integration backend-agnostic. Chroma still deferred (needs SQLCipher). See §4.5. |
| **5.5** | Revocation cost UX | **Both soft + hard, soft default, `--rotate` for hard** | `revoke` = manifest mark, fast, doesn't invalidate cached bytes (loud warning). `revoke --rotate` = DEK rotate + re-encrypt; cached bytes become useless. See §4.3. |
| **5.6** | Migration approach | **In-place `axon project seal`** | Atomic per-file (`.sealing` sibling + fsync + rename); recovery scanner for crashed migrations. NOT force re-ingest — preserves user's RAPTOR / GraphRAG / dynamic-graph state. |

**Knock-on effects for the phase plan (§6 below):**

- **Phase 2 grows** to cover both TQDB and LanceDB integration. Same
  PR — both backends share the cache layer so the diff is mostly
  one-time work plus a thin per-backend mount hook each.
- **Phase 4 (revocation) splits into two flows**: `revoke` (cheap,
  manifest-only) and `revoke --rotate` (expensive, full project
  re-encrypt). Each needs its own progress UX + audit-log entry.
- **The "ephemeral cache" subsystem becomes a Phase 2 building
  block** — design it before the backend integration so both TQDB
  and LanceDB can reuse it.

## 6. Phased implementation

Each phase = one focused PR. Phases 1–2 are the MVP that delivers
end-to-end encryption for a single backend (TQDB) on a single owner;
later phases add reach and polish.

| Phase | Deliverable | Status |
|---|---|---|
| **0** | This doc + §5 decisions locked + per-phase GitHub issues opened | ✅ doc landed, decisions locked 2026-04-25 |
| **1** | Crypto foundations: `axon.security.crypto` module (`SealedFile.{write,read}`, AES-KW wrap/unwrap, KEK derivation, `make_aad`), keyring integration with `KeyringUnavailableError`, tests on synthetic data. **Zero behavior change for existing projects.** | ✅ **SHIPPED** — PR #56 / `feat/sealed-mount-phase-1` (2026-04-25) |
| **2** | **Ephemeral cache subsystem** + **TQDB sealed read** + **LanceDB sealed read** — owner can `axon project seal <name>` on either backend; mount flow reads from the cache; cache wiped on close; PID-based crash recovery. Both backends in one PR (cache makes them backend-agnostic). | ✅ **SHIPPED** (`cache.py` + `seal.py` + `mount.py` + `main.py` integration) |
| **3** | Sealed-share generation + redemption flow — fill in `generate_sealed_share` / `redeem_sealed_share` stubs; grantee can query a sealed project on a shared filesystem | ✅ **SHIPPED** (`share.py` `generate_sealed_share` / `redeem_sealed_share`) |
| **4** | Revocation: soft `revoke` (manifest mark) + hard `revoke --rotate` (re-encrypt + per-share KEK regeneration); progress UX for hard; tests covering both flows + cached-bytes-after-rotate negative case | ✅ **SHIPPED** (`share.py` `revoke_sealed_share`, PR #75) |
| **5** | Cross-interface surfaces — REST `/store/seal`, `/share/generate?sealed=true&ttl_days=N`, `/share/revoke?rotate=true`; MCP tools (`seal_project`, `share_project sealed=true`, `revoke_share rotate=true`); REPL `/store seal <name>`, `/share generate --sealed`, `/share revoke --rotate`; CLI flags; docs (`SHARE_MOUNT.md` rewrite + cross-link) | ✅ **SHIPPED** |
| **6** | Passphrase fallback for headless / no-keyring environments (Phase 1 deferred from §5.3) | ✅ **SHIPPED** |
| **7** | Verification: extend `scripts/qa/SEALED_SHARE_SMOKE.md` with sealed-store steps; real two-machine OneDrive run before tagging | ✅ **SHIPPED** (`scripts/qa/SEALED_SHARE_SMOKE.md` §"Phase 7" + `tests/test_sealed_switch_project.py`) |

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

Those go in **`scripts/qa/SEALED_SHARE_SMOKE.md`** — extended with sealed-store
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
- **Multi-owner / write access for grantees.** Mounts stay read-only
  by design.
- **Key escrow / recovery** (e.g. Shamir-secret-sharing the master
  key across N admins). Lose master = lose project. Document, don't
  fix.

## 10. Pre-implementation checklist

- [x] §5 decisions confirmed (locked 2026-04-25)
- [x] Add `cryptography` and `keyring` to `pyproject.toml` `[project.optional-dependencies]` under a new `sealed` extra (Phase 1 / PR #56)
- [x] Settle naming: "sealed share" — current code uses it, this plan follows
- [x] `src/axon/security/` is the canonical home (Phase 1 converted `security.py` to a package)
- [x] AAD content for AES-GCM = `key_id || NUL || file_relpath` via `make_aad()` (Phase 1)
- [ ] Open GitHub issue for Phase 2 (cache subsystem + TQDB + LanceDB sealed read paths)
- [ ] Open GitHub issues for Phases 3–7
