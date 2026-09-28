# Sharing Knowledge Bases

## Overview

Axon supports two sharing modes:

- **Plaintext** — fastest setup, works on local disk and on-premises SMB3/NFS. Do NOT use with cloud sync drives.
- **Sealed (encrypted-at-rest)** — safe for OneDrive, Dropbox, and Google Drive. All project files are AES-256-GCM encrypted; cloud providers see only ciphertext.

Choose based on how the shared folder reaches both machines:

| Sync method | Mode to use |
|---|---|
| Local disk / USB | Plaintext or Sealed |
| On-premises SMB3 / NFS | Plaintext or Sealed |
| OneDrive Personal/Business | **Sealed only** |
| Dropbox | **Sealed only** |
| Google Drive Desktop (Mirror mode) | **Sealed only** |
| Google Drive Desktop (Stream mode) | Not supported (evicts files mid-query) |

---

## Plaintext Sharing

### What it is

In plaintext mode, Axon reads the owner's project files directly from the shared path. A grantee mounts the project by redeeming a share key — Axon copies no data itself; the filesystem delivers the bytes. This works well on local disk or a properly-configured SMB3 share where file locking and atomic renames work correctly.

### Why OneDrive is unsafe for plaintext mode

SQLite-backed components write `-wal` and `-shm` sidecar files. Cloud sync clients (OneDrive, Dropbox, Google Drive) reorder these sidecars during upload, producing a corrupt database on the receiving machine. Conflict-copy filenames are silently ignored by Axon. See the [Filesystem Compatibility Matrix](#filesystem-compatibility-matrix) below for the full list of supported and unsupported filesystems.

### Owner setup (4 steps)

Step 1: Initialise the store at a shared path

```bash
axon --store-init /path/to/shared/AxonStore
```

Step 2: Create a project and ingest content

```bash
axon --project-new research
axon --project research --ingest /path/to/documents
```

Step 3: Generate a share key for the grantee

```bash
axon --share-generate research alice
```

Axon prints a share string and a key ID. Transmit the share string to the grantee out-of-band (Slack, email, etc.).

To set a time limit on the share:

```
axon> /share generate research alice --ttl-days 30
```

`--ttl-days` is honoured for both plaintext and sealed shares as of v0.4.0 (plaintext-only in v0.3.x). For plaintext shares, the expiry is recorded in the share manifest and enforced by the grantee's client on the next switch or query. To renew or clear a plaintext expiry without re-issuing the share, use `axon --share-extend <key_id> --share-ttl-days N` (renew) or `axon --share-extend <key_id>` (omit `--share-ttl-days` to clear); the REPL equivalent is `/share extend <key_id> --ttl-days N` or `/share extend <key_id> --clear`. `ttl_days <= 0` is rejected — there's no `--ttl-days 0` shortcut for clearing. Sealed shares cannot be extended in place — see [Renewing a sealed share](#renewing-a-sealed-share) below.

Step 4 (grantee): Redeem the share

```bash
axon --share-redeem "<paste share string>"
axon --project mounts/owner_research
```

### Revoking a plaintext share

```bash
axon --share-revoke <key_id>
```

This marks the share as revoked in the owner's share manifest. The grantee is refused on their next project switch and on their next query — every surface (CLI, REPL, REST, MCP, VS Code) runs the same check — and their mount entry is removed the next time they list shares or projects. See [How share validity is decided](#how-share-validity-is-decided).

---

## Sealed Sharing (OneDrive / Dropbox / Google Drive)

### What is sealed sharing?

Sealed sharing encrypts every project content file in place using AES-256-GCM with a per-project Data Encryption Key (DEK). The DEK is wrapped (encrypted) separately for each grantee using key material derived from the share token; only a holder of the matching share token can unwrap the DEK and read the files. Cloud providers see only ciphertext — they cannot read your documents even if they have full access to the folder.

When a grantee queries a sealed project, Axon decrypts the files into an ephemeral cache in the OS temp directory, points the query engine at the cache, and securely wipes the cache on exit.

### Prerequisites

- Both machines need sealed-share libraries (`cryptography` + `keyring`). These are bundled in the recommended `[starter]` install — `pip install "axon-rag[starter]"`. If you skipped the bundle, install just the sealed extra: `pip install "axon-rag[sealed]"`.
- A shared sync folder accessible to both owner and grantee (OneDrive, Dropbox, Google Drive Mirror mode)
- The folder must be set to sync fully to disk on both machines:
  - **OneDrive**: right-click the folder in Explorer → "Always keep on this device" (disables Files On-Demand for that folder)
  - **Google Drive**: use Mirror mode, not Stream mode (set in Google Drive preferences → Mirror files)
  - **Dropbox**: Selective Sync must include the folder (right-click → "Make available offline")

### Owner setup (5 steps)

Step 1: Initialise the store at the shared sync folder

```bash
axon --store-init "/path/to/OneDrive/AxonStore"
```

Step 2: Bootstrap the sealed store (first time only on this machine)

```bash
axon --store-bootstrap "your-passphrase"
```

This generates a master key and persists it in the OS keyring (DPAPI on Windows, Keychain on macOS, Secret Service on Linux). The passphrase is used as a recovery fallback for headless environments. Losing the passphrase means losing access to every sealed project on this machine — write it down and store it securely.

Step 3: Create a project and ingest content

```bash
axon --project-new research
axon --project research --ingest /path/to/documents
```

Step 4: Seal the project

```bash
axon --store-unlock          # required after every process restart before --project-seal works
axon --project-seal research
```

This encrypts all content files (meta.json, BM25 index, vector store files) in place. Each file is rewritten atomically: Axon writes to a `.sealing` sidecar, syncs to disk, then renames over the original. `version.json` stays plaintext so grantees can detect when the owner has re-ingested without needing the DEK. Takes roughly 1 second per 100 MB of project data.

Step 5: Generate a sealed share for the grantee

```bash
axon --share-generate research alice
```

Axon detects that the project is sealed and automatically generates a sealed share (key ID starts with `ssk_`). Send the share string to the grantee out-of-band.

#### SEALED1 vs SEALED2 envelopes

| Envelope | Decoded payload shape | Role in v0.4.0+ |
|---|---|---|
| `SEALED1:` | `SEALED1:<key_id>:<token_hex>:<owner>:<project>:<store_path>` (6 colon-separated fields including the prefix) | Legacy redeem-only — accepted for pre-v0.4.0 share strings, never minted by current generators. Mounts succeed but TTL cannot be enforced (no signing key bound to the envelope). |
| `SEALED2:` | `SEALED2:<key_id>:<token_hex>:<owner>:<project>:<store_path>:<pubkey_hex_64>` (7 fields; the trailing pubkey is split first via `rpartition(":")` so a Windows `store_path` containing `C:\...` parses correctly) | Default in v0.4.0+. The 7th field carries the owner's Ed25519 signing pubkey, derived deterministically from the master via `HKDF-SHA256(master, info=b"axon-share-signing-v1")`. The grantee uses this pubkey to verify the expiry sidecar on every mount. |

#### Sealed-share TTL flow (v0.4.0)

To add an expiry, pass `--ttl-days` at generation time:

```
axon> /share generate research alice --ttl-days 30
```

The end-to-end lifecycle:

1. **Generate (owner).** Axon mints a SEALED2 envelope and writes two files to `<project>/.security/shares/`:
   - `<key_id>.wrapped` — the wrapped DEK (~40 bytes).
   - `<key_id>.expiry` — an Ed25519-signed JSON sidecar (~250 bytes) with the shape:
     ```json
     {
       "key_id": "ssk_a1b2c3d4",
       "expires_at": "2026-06-01T17:30:00Z",
       "sig": "<base64url-encoded Ed25519 signature over b'<key_id>:<expires_at>'>"
     }
     ```
     The signature covers the bytes `b"{key_id}:{expires_at}"` exactly. The signing key is derived deterministically from the owner's master via HKDF-SHA256 with `info=b"axon-share-signing-v1"` and a fixed non-empty salt — see `axon.security.signing._SIGNING_SALT` and `SIGNING_HKDF_INFO` for the canonical constants. Wait for **both** files to finish uploading before telling the grantee to redeem.
2. **Redeem (grantee).** The SEALED2 envelope's 7th field carries the owner's signing pubkey. Axon stores the pubkey in the mount descriptor (`<user_dir>/mounts/<mount_name>/mount.json`) under `owner_pubkey_hex` so it survives across process restarts.
3. **Mount (grantee).** On every `switch_project` to a sealed mount, Axon reads the mount descriptor's pubkey, fetches the latest `<key_id>.expiry` from the synced FS, verifies the Ed25519 signature against the recorded pubkey, parses `expires_at`, and compares against `datetime.now(timezone.utc)`. Any of these conditions raises `ShareExpiredError`:
   - sidecar JSON is malformed or missing required fields,
   - signature verification fails (tampered or wrong key),
   - `key_id` inside the sidecar doesn't match the share,
   - `expires_at` is naive / not valid ISO 8601,
   - `now > expires_at`,
   - mount descriptor has no `owner_pubkey_hex` (legacy SEALED1 mount — TTL is unenforceable; the sidecar is treated as untrusted and the mount fails closed).
4. **Auto-destroy (grantee).** When `ShareExpiredError` fires, Axon performs three local cleanups:
   - **DEK** — deletes from the OS keyring (`axon.share.<key_id>`) and the file fallback at `<user_dir>/.security/shares/<key_id>.dek.wrapped`.
   - **Plaintext cache** — wipes the active ephemeral cache directory (the temp dir staged for an earlier mount).
   - **Mount descriptor** — removes `<user_dir>/mounts/<mount_name>/mount.json`, so the project disappears from `list_projects` immediately.

   **Encrypted source files on the synced filesystem are NEVER touched.** Deleting them would propagate the deletion back to the owner via OneDrive/Dropbox/Drive sync — a destructive failure mode. The owner manages their own files; the grantee only manages local state.

If the grantee's clock is behind the owner's by more than the TTL, expiry won't fire on the grantee yet — accept that the comparison is grantee-local. If the grantee's clock is ahead, expiry fires early; the grantee can re-redeem a fresh share to recover.

#### Renewing a sealed share

`POST /share/extend` (and the equivalent `axon --share-extend` / `/share extend` REPL command) only operates on the **plaintext** share manifest. Sealed shares (`ssk_*` key IDs) are not present in that manifest, so calling `/share/extend ssk_...` returns **404 Not Found** by design. To renew a sealed share:

1. Generate a fresh sealed share with the new expiry: `/share generate <project> <grantee> --ttl-days N`.
2. Send the new SEALED2 string to the grantee.
3. After the grantee has redeemed and mounted successfully, revoke the old share: `/share revoke <old_key_id> --project <project>`. Use `--rotate` if the old grantee machine is compromised; soft revoke is sufficient if you simply want to retire the old key ID.

### Grantee setup (3 steps)

Step 1: Initialise the store pointing to the same shared folder

```bash
axon --store-init "/path/to/OneDrive/AxonStore"
```

Step 2: Redeem the sealed share

```bash
axon --share-redeem "SEALED2:..."
```

Axon reads the wrapped DEK from the shared folder, derives the decryption key from the share token, and stores the unwrapped DEK in the OS keyring. The share token is discarded from process memory immediately. The redeem path also extracts the owner's Ed25519 signing pubkey from the SEALED2 envelope (7th field) and persists it in the mount descriptor so subsequent mounts can verify the expiry sidecar (see [TTL flow](#sealed-share-ttl-flow-v040) above). Pre-v0.4.0 `SEALED1:` strings still redeem successfully but mount without TTL enforcement — TTL gating requires a SEALED2 envelope to bind the signing key cryptographically to the share.

Step 3: Query the sealed project

```bash
axon --project mounts/owner_research "your question"
```

Or in the REPL:

```
axon> /project mounts/owner_research
axon> What does the corpus say about X?
```

Axon decrypts the project into a secure temp folder, runs the query, and wipes the temp folder on exit.

### Revoking sealed access

**Soft revoke** (fast, does not invalidate a grantee's cached DEK):

```bash
axon --share-revoke ssk_abc123 --share-project research
```

Or in the REPL:

```
axon> /share revoke ssk_abc123 --project research
```

Deletes the share's `.wrapped` file (plus its `.kek` and `.expiry` sidecars) so the share string cannot be redeemed again. Once that deletion has synced to the grantee, Axon refuses the mount on their next switch or query and removes the mount entry on their next share/project listing. The grantee's **cached DEK is not deleted** from their OS keyring, though — soft revoke is enforced by the grantee's Axon client, so a grantee running modified software could still decrypt the ciphertext they already have. Use soft revoke when the grantee is cooperative or has simply lost access to the machine; use hard revoke when you need to guarantee termination of access.

**Hard revoke** (slow, re-encrypts everything with a new DEK):

```bash
axon --share-revoke ssk_abc123 --share-project research --share-rotate
```

Or in the REPL:

```
axon> /share revoke ssk_abc123 --project research --rotate
```

Generates a new DEK, re-encrypts every content file in the project, and selectively re-wraps the new DEK for surviving grantees that have a per-share KEK sidecar (`.security/shares/<key_id>.kek`). The revoked grantee's cached DEK no longer matches the ciphertext; their next query fails with an authentication error. Surviving grantees on legacy projects (predating per-share KEK persistence) are also invalidated and must redeem fresh shares — both the CLI flag (`axon --share-revoke ... --share-rotate`) and the REPL command (`/share revoke ... --rotate`) return an `invalidated_share_key_ids` list so you know who to re-issue. Use when the grantee machine is compromised or you suspect unauthorized access.

Note: hard revoke re-encrypts all N files, so it takes roughly as long as the original seal. Wait for the sync folder to upload all re-encrypted files before the revoke takes effect on the grantee side.

**List current shares:**

```bash
axon --share-list
```

### Security properties

| Threat | Protected? |
|---|---|
| Cloud provider reads your files (OneDrive, Dropbox, etc.) | Yes — AES-256-GCM ciphertext only |
| Another OneDrive collaborator reads your project | Yes — they do not have the decryption key |
| Grantee continues querying after soft revoke | Partial — Axon refuses the mount once the wrap deletion has synced, but the cached DEK stays in the grantee's OS keyring, so modified client software could still decrypt; use hard revoke for a cryptographic guarantee |
| Grantee continues querying after hard revoke | Yes — new DEK, old cached key fails with authentication error |
| Unencrypted temp files on grantee disk during a query | Partial — ephemeral cache lives in OS temp dir during the query session, wiped securely on exit |

### Troubleshooting sealed sharing

| Error | Cause | Fix |
|---|---|---|
| `SecurityError: Store is locked` | Master key is not loaded | Run `axon --store-unlock <passphrase>` first |
| `SecurityError: Project DEK file missing` | Project is not sealed, or DEK file was not synced | Run `axon --project-seal <name>` if owner; wait for sync if grantee |
| `CacheCapacityError: Not enough disk space` | Temp dir needs at least 1.1× the project size free | Free space in OS temp dir or set `TMPDIR` to a larger volume |
| `SecurityError: Wrapped DEK won't unwrap` / `InvalidTag` | Hard revoke was performed — grantee's cached DEK is stale | Owner must generate a new share; grantee redeems it |
| `... has been revoked by the owner (share wrap removed)` | The owner soft- or hard-revoked this share | Ask the owner for a new share |
| `... cannot be verified` (state `unverifiable`) | The owner's project, share directory, sealed marker or `.expiry` file is not readable yet (offline, sync incomplete, cloud placeholder) | Wait for sync / make the folder "Always keep on this device", then retry — the mount is kept, nothing is deleted |
| `Share '<project>' (key <id>) expired at <ts>` (state `expired`) | The expiry sidecar's `expires_at` has passed | Owner generates a fresh share with a new `--ttl-days`; grantee redeems. Sealed shares cannot be extended in place — see [Renewing a sealed share](#renewing-a-sealed-share). |
| `... failed signature or format verification; treating the share as expired` (reason `expiry_unverified`) | Sidecar tampered, owner pubkey rotated, or wrong sidecar synced | Owner regenerates the share; grantee redeems. Local DEK + cache + mount descriptor are auto-destroyed; encrypted source files on the synced FS are untouched. |
| `404 Not Found` from `POST /share/extend` on a sealed key | Sealed shares (`ssk_*`) are not in the plaintext manifest | Mint a fresh sealed share with `--ttl-days` and revoke the old one |
| OneDrive shows "Files On-Demand" cloud icons on project files | Files are placeholders and will fail mid-query | Right-click the project folder → "Always keep on this device" |
| Google Drive Stream mode evicts files | Stream mode removes cached files to free disk space | Switch to Mirror mode in Google Drive preferences |

---

## How share validity is decided

Every surface — `axon --project mounts/...`, REPL `/project switch`, REST `/project/switch`, MCP, VS Code, the per-query check that runs before every retrieval, `@mounts`/`@store` scopes, and the share/project listings — asks one function (`axon.share_validity.share_status`) and gets the same answer. No extra record is kept for this: the authority is the owner-side file that already exists for each share.

| Share kind | Authoritative record (on the owner's synced store) |
|---|---|
| Plaintext (`sk_*`) | The key's entry in `<owner>/.shares/.share_manifest.json` (`revoked`, `expires_at`) |
| Sealed (`ssk_*`) | `<project>/.security/shares/<key_id>.wrapped` exists (deleted on revoke) + the owner-signed `<key_id>.expiry` sidecar, if any |

The grantee's `mounts/<name>/mount.json` and the `received` entries in `.share_keys.json` are only pointers to those records.

| State | Typical `reason` | Access | What happens to the grantee's mount |
|---|---|---|---|
| `valid` | `ok` | allowed | kept |
| `revoked` | `revoked` (plain), `wrap_absent` (sealed) | denied | mount entry removed on the next listing; a sealed share's cached DEK is **kept** |
| `expired` | `expired`, `expiry_unverified` (sealed sidecar tampered/malformed) | denied | mount entry removed; a sealed share's cached DEK is **deleted** (auto-destroy) |
| `unverifiable` | `manifest_unreadable`, `record_absent`, `target_missing`, `shares_dir_missing`, `sealed_marker_missing`, `expiry_unreadable` | denied | **kept** — nothing is deleted |
| `invalid` | `key_id_missing`, `target_unset`, `descriptor_inactive`, … | denied | kept (the local descriptor itself is malformed; re-redeem) |

**Offline / sync rule:** if the owner's record cannot be read — the owner's store is offline, a sync is incomplete, or the file is a cloud placeholder that fails to open — the share is `unverifiable`: access is refused until the record is readable again, but the mount is never deleted, so it comes back by itself once sync catches up. Only a record that positively says *revoked* or *expired* removes anything. Plaintext expiry keeps a 5-minute clock-skew allowance; sealed expiry is checked strictly against the signed timestamp.

`GET /projects` (`shared_mounts`), `GET /share/list` and `list_share_mounts()` report the decision as additive `state` / `reason` fields; `is_broken` is simply "not `valid`". The owner-side view uses the same records: `POST /project/delete` is blocked only by shares that are still `valid` (plaintext or sealed) — revoked or expired shares no longer block deletion. The owner's sealed view reads the `.expiry` timestamp without verifying its signature (that needs the unlocked master key).

A corrupt `.share_manifest.json` or `.share_keys.json` is never silently rewritten: generate / revoke / extend refuse with a "refusing to overwrite" error (REST: `409`) naming the file, so revocation records can't be lost to a truncated sync.

---

## Moving a Sealed Project to a Different OS

Sealed project files are AES-256-GCM ciphertext — the format is identical on every
platform (big-endian header, standard nonce/tag layout, forward-slash paths in AAD).
After decrypting into the temp cache, LanceDB (Apache Arrow), BM25 (msgpack/JSON),
and TurboQuantDB (manifest.json) all open correctly on any OS.

The only thing you need to carry across is the **master key** so Axon can unwrap
the project DEK on the destination machine.

### Steps

1. **Copy the sealed project directory** to the same relative location on the new machine,
   or point `store.base` in `config.yaml` to the shared sync folder.

2. **Copy `master.enc`** from the source machine:
   ```
   <store_base>/<owner>/.security/master.enc
   ```
   Place it at the same relative path on the destination.

   > Axon now always writes this file alongside the OS keyring (DPAPI / Keychain /
   > Secret Service), so it exists on every platform after the first `--store-bootstrap`.

3. **Unlock on the destination** with the same passphrase:
   ```
   axon --store-unlock
   ```
   Axon checks the OS keyring first (no entry on a fresh machine), then falls back to
   `master.enc` automatically. No extra flags are needed.

4. **Switch to the project and query normally:**
   ```
   axon --project research
   axon> What are the key findings?
   ```

### Security note

`master.enc` is protected by your passphrase via scrypt (N=2¹⁵). Someone who has the
file but not the passphrase cannot decrypt anything. Treat it with the same care as a
password manager export — store it only in locations you control.

---

## Filesystem Compatibility Matrix

This matrix applies to **plaintext sharing** only. Sealed sharing works through any filesystem because the content is encrypted before it reaches the filesystem.

| Filesystem | Verdict | Notes |
|---|---|---|
| **Local disk** (NTFS / ext4 / APFS / ZFS) | Safe | Single-writer owner, many-reader grantees on the same machine works out of the box. |
| **On-premises SMB3 on Windows Server 2019+** | Safe, with caveats | Grantees must be Windows-native (not WSL). SMB3 leases give the reader a coherent view. Avoid SQLite-backed components on the share. |
| **DFS Namespace (DFS-N, without DFS-R)** | Thin alias | Pure referral layer over a single SMB server. Inherits whatever that SMB share gives you. |
| **Azure Files (SMB3.1.1)** | Usable for reads | Continuous-Availability retry window hangs clients for minutes during drops. Keep SQLite-backed state (e.g. `.dynamic_graph.db`) on local disk. |
| **OneDrive** (Personal / Business / SharePoint) | Unsafe for plaintext | Files On-Demand placeholders hang `mmap`; `-wal`/`-shm` sidecars sync out of order and corrupt SQLite; conflict copies silently ignored. Use sealed mode instead. |
| **Dropbox** (Personal / Business) | Unsafe for plaintext | Same SQLite sidecar reorder; conflicted copy files silently ignored. Use sealed mode instead. |
| **Google Drive for Desktop** (Mirror or Stream) | Unsafe for plaintext | Same SQLite corruption; `.tmp.drivedownload` clutter; Stream mode cache evicts mid-query. Use sealed mode instead. |
| **DFS Replication (DFS-R)** | Unsafe | `ConflictAndDeleted` silently eats "losing" index files; 15-minute minimum replication interval. |
| **WebDAV redirector** | Unsafe | 50 MB default file cap; directory caches serve stale reads. |
| **WSL + Windows mount** (`/mnt/c/...`) | Unsafe for owner | WSL1 `fcntl(F_SETLK)` is broken; WSL2 cifs emulation is unpredictable. Symptom: `"attempt to write a readonly database"`. |

### Recommended backend per supported filesystem

| Filesystem | Recommended backend | Rationale |
|---|---|---|
| Local disk | TurboQuantDB (default) | Single-binary mmap, best recall/size tradeoff. |
| SMB3 / DFS-N | TurboQuantDB or LanceDB | TurboQuantDB for small/medium corpora; LanceDB's immutable fragments replicate cleanly when the owner compacts on a schedule. |
| Azure Files | TurboQuantDB | Keep SQLite-backed state (e.g. `.dynamic_graph.db`) off the share. |

Do not use Chroma on any shared, network, or cloud-sync path. Chroma is also not supported in sealed mode (requires SQLCipher — a separate, larger project).

---

## Security Considerations

### Windows users with highly sensitive data

The encrypted cache is wiped with random-byte overwrite on close, but NTFS copy-on-write and SSD TRIM/wear-leveling may retain plaintext in freed sectors even after the overwrite completes. On Windows, Axon calls `FlushFileBuffers` (Windows API) after each file wipe for best-effort write-through, but this does not guarantee physical sector erasure on SSDs or eliminate NTFS copy-on-write remnants.

For compliance or classified use, take one of the following measures:

- **Enable BitLocker** on the drive containing `%TEMP%` (the default cache location). BitLocker encrypts every sector, so even previously-freed sectors are protected at rest.
- **Redirect the cache to an encrypted volume** by setting the `AXON_CACHE_DIR` environment variable to a path on a BitLocker drive, a VeraCrypt container, or a RAM disk:
  ```
  set AXON_CACHE_DIR=E:\secure-tmp
  ```
- **Use a RAM disk** (e.g. ImDisk, RamMap) for `%TEMP%`. A RAM disk never writes to physical storage; cache contents are destroyed when the machine powers off.

Axon logs an INFO-level reminder (`SealedCache: Windows NTFS secure-delete is best-effort`) when it first wipes a cache directory on Windows. This reminder is emitted once per cache directory per process lifetime.

### Passphrase strength

`axon --store-bootstrap` and `axon --store-change-passphrase` require a passphrase of at least 8 characters. This is a minimum floor; a stronger passphrase significantly increases resistance to offline brute-force attacks against the scrypt-wrapped master key (N=2¹⁵, r=8, p=1). A 16+ character random passphrase or a four-word diceware phrase is recommended.

`axon --store-unlock` does not enforce the minimum length so it can distinguish "wrong passphrase" from "too short" without misleading error messages.

---

## See Also

- [ADMIN_REFERENCE.md](ADMIN_REFERENCE.md) — complete command reference for all share and seal operations
- [QUICKREF.md](QUICKREF.md) — command cheat sheet
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — broader error patterns
- [AXON_STORE.md](AXON_STORE.md) — how AxonStore projects and mounts work
- [docs/architecture/SEALED_SHARING_DESIGN.md](architecture/SEALED_SHARING_DESIGN.md) — technical design, key hierarchy, and threat model for sealed sharing
