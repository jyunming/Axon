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

Step 4 (grantee): Redeem the share

```bash
axon --share-redeem "<paste share string>"
axon --project mounts/owner_research
```

### Revoking a plaintext share

```bash
axon --share-revoke <key_id>
```

This marks the share as revoked in the manifest. The grantee loses access on the next switch or query.

---

## Sealed Sharing (OneDrive / Dropbox / Google Drive)

### What is sealed sharing?

Sealed sharing encrypts every project content file in place using AES-256-GCM with a per-project Data Encryption Key (DEK). The DEK is wrapped (encrypted) separately for each grantee using key material derived from the share token; only a holder of the matching share token can unwrap the DEK and read the files. Cloud providers see only ciphertext — they cannot read your documents even if they have full access to the folder.

When a grantee queries a sealed project, Axon decrypts the files into an ephemeral cache in the OS temp directory, points the query engine at the cache, and securely wipes the cache on exit.

### Prerequisites

- Both machines need the `[sealed]` extra: `pip install "axon-rag[sealed]"` (installs `cryptography` and `keyring`)
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
axon --project-seal research
```

This encrypts all content files (meta.json, BM25 index, vector store files) in place. Each file is rewritten atomically: Axon writes to a `.sealing` sidecar, syncs to disk, then renames over the original. `version.json` stays plaintext so grantees can detect when the owner has re-ingested without needing the DEK. Takes roughly 1 second per 100 MB of project data.

Step 5: Generate a sealed share for the grantee

```bash
axon --share-generate research alice
```

Axon detects that the project is sealed and automatically generates a sealed share (key ID starts with `ssk_`). The printed share string begins with `SEALED1:`. Send it to the grantee out-of-band.

To add an expiry:

```
axon> /share generate research alice --ttl-days 30
```

Wait for the sync folder to upload the new `.security/shares/<key_id>.wrapped` file (about 40 bytes) to the cloud before telling the grantee to redeem.

### Grantee setup (3 steps)

Step 1: Initialise the store pointing to the same shared folder

```bash
axon --store-init "/path/to/OneDrive/AxonStore"
```

Step 2: Redeem the sealed share

```bash
axon --share-redeem "SEALED1:..."
```

Axon reads the wrapped DEK from the shared folder, derives the decryption key from the share token, and stores the unwrapped DEK in the OS keyring. The share token is discarded from process memory immediately.

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

Marks the share as revoked and deletes the `.wrapped` file so the share string cannot be redeemed again. A grantee who has already redeemed still has the DEK cached in their OS keyring — soft revoke blocks new redemptions but does not remove the cached key, so a cooperative grantee can continue querying indefinitely. Use when the grantee is cooperative or has simply lost access to the machine; use hard revoke when you need to guarantee termination of access.

**Hard revoke** (slow, re-encrypts everything with a new DEK):

```bash
axon --share-revoke ssk_abc123 --share-project research --share-rotate
```

Or in the REPL:

```
axon> /share revoke ssk_abc123 --project research --rotate
```

Generates a new DEK, re-encrypts every content file in the project, and deletes all existing share wraps (including wraps for other active grantees — they must redeem new shares). The grantee's cached DEK no longer matches the ciphertext; their next query fails with an authentication error. Use when the grantee machine is compromised or you suspect unauthorized access.

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
| Grantee continues querying after soft revoke | Partial — cached DEK in OS keyring remains valid indefinitely; soft revoke only blocks new redemptions |
| Grantee continues querying after hard revoke | Yes — new DEK, old cached key fails with authentication error |
| Unencrypted temp files on grantee disk during a query | Partial — ephemeral cache lives in OS temp dir during the query session, wiped securely on exit |

### Troubleshooting sealed sharing

| Error | Cause | Fix |
|---|---|---|
| `SecurityError: Store is locked` | Master key is not loaded | Run `axon --store-unlock <passphrase>` first |
| `SecurityError: Project DEK file missing` | Project is not sealed, or DEK file was not synced | Run `axon --project-seal <name>` if owner; wait for sync if grantee |
| `CacheCapacityError: Not enough disk space` | Temp dir needs at least 1.1× the project size free | Free space in OS temp dir or set `TMPDIR` to a larger volume |
| `SecurityError: Wrapped DEK won't unwrap` / `InvalidTag` | Hard revoke was performed — grantee's cached DEK is stale | Owner must generate a new share; grantee redeems it |
| `Sealed-share wrap file missing` | Owner revoked or the sync has not delivered the wrap file yet | Wait for sync to complete, then retry |
| OneDrive shows "Files On-Demand" cloud icons on project files | Files are placeholders and will fail mid-query | Right-click the project folder → "Always keep on this device" |
| Google Drive Stream mode evicts files | Stream mode removes cached files to free disk space | Switch to Mirror mode in Google Drive preferences |

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
| **Azure Files (SMB3.1.1)** | Usable for reads | Continuous-Availability retry window hangs clients for minutes during drops. Keep `.governance.db` on local disk. |
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
| Azure Files | TurboQuantDB | Keep `.governance.db` off the share. |

Do not use Chroma on any shared, network, or cloud-sync path. Chroma is also not supported in sealed mode (requires SQLCipher — a separate, larger project).

---

## Security Considerations

### Windows users with highly sensitive data

The encrypted cache is wiped with random-byte overwrite on close, but NTFS copy-on-write and SSD wear-leveling may retain plaintext in freed sectors even after the overwrite completes. On a spinning HDD, Axon also calls `FlushFileBuffers` (Windows API) after each file wipe for best-effort write-through, but this does not guarantee physical sector erasure on SSDs.

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
