# Sharing Knowledge Bases

Axon shares a project with another person as a **read-only mount**: you issue a share
string, they redeem it, and your project appears in their project list as
`mounts/<you>_<project>`. There is no server in between — both machines read the same
folder. Every command below is also on the REST API, and all but hard revocation are MCP
and VS Code tools too ([REST](REFERENCE.md#86-sharing-bodies),
[MCP](REFERENCE.md#93-tools)).

Two modes:

- **Plaintext** — the simplest setup; for local disk and on-premises SMB3 / NFS. **Not** for
  cloud sync drives.
- **Sealed** — every project file is encrypted at rest with AES-256-GCM; safe on OneDrive,
  Dropbox and Google Drive, which only ever see ciphertext.

| How the folder reaches both people | Mode |
|---|---|
| Same machine, local disk, USB | Plaintext or sealed |
| On-premises SMB3 / NFS | Plaintext or sealed |
| OneDrive Personal / Business | **Sealed only** |
| Dropbox | **Sealed only** |
| Google Drive for Desktop, Mirror mode | **Sealed only** |
| Google Drive for Desktop, Stream mode | Not supported (evicts files mid-query) |

---

## The store

Every user's data lives under a **store base**, in `<base>/AxonStore/<username>/`. The
default base is `~/.axon`, which is private to you. To share, both people point their
store at the same folder:

```bash
axon --store-init /srv/shared/axon      # data moves to /srv/shared/axon/AxonStore/<you>/
axon --store-whoami                     # your username, store path and user directory
```

`--store-init` writes `store.base` to your `config.yaml` (REPL: `/store init <path>`;
REST: `POST /store/init {"base_path": ..., "persist": true}`; VS Code: *Axon: Initialize
AxonStore*). You can also set `store.base` in `config.yaml` or the `AXON_STORE_BASE`
environment variable. Moving the base makes projects at the old base unreachable until you
move back — the REST response lists them under `unreachable_projects`.

Your username is your OS login name. Each user's projects are theirs alone; another user
reads them only through a share. Shares are always read-only: ingest, delete, clear and
fact updates on a mount return `403` / `PermissionError`.

**Key ids and strings.** Plaintext shares have key ids `sk_…`, sealed shares `ssk_…`. The
share string is a base64 blob; send it out of band (Signal, encrypted email — not the same
channel as the data). A mount of `alice`'s project `research/papers` appears to the grantee
as `mounts/alice_research_papers` (`/` becomes `_`).

---

## Plaintext Sharing

In plaintext mode the grantee's Axon reads the owner's project files directly from the
shared folder. That works where file locking and atomic renames behave — local disk and
properly configured SMB3. Cloud sync clients reorder SQLite's `-wal` / `-shm` sidecars
and binary index writes during upload, which corrupts the copy on the other machine; see
the [compatibility matrix](#filesystem-compatibility-matrix).

### Owner

```bash
axon --store-init /srv/shared/axon
axon --project-new research --ingest /path/to/documents
axon --share-generate research alice                    # prints the share string and key id
axon --share-generate research alice --share-ttl-days 30   # …or with an expiry
```

REPL: `/share generate research alice --ttl-days 30`. The expiry is stored in the owner's
share manifest and enforced by the grantee's Axon on every switch and query. Renew or clear
it without re-issuing:

```bash
axon --share-extend sk_a1b2c3d4 --share-ttl-days 30     # new expiry, 30 days from now
axon --share-extend sk_a1b2c3d4                         # no expiry
```

REPL: `/share extend <key_id> --ttl-days N` or `--clear`. `--ttl-days 0` is rejected.

### Grantee

```bash
axon --store-init /srv/shared/axon                      # the same shared base
axon --share-redeem "<share string>"
axon --project mounts/owner_research "What are the key findings?"
```

In the REPL: `/share redeem <share string>`, then `/project switch mounts/owner_research`.

### Revoking

```bash
axon --share-revoke sk_a1b2c3d4
```

The key is marked revoked in the owner's manifest. The grantee is refused on their next
switch or query — every surface runs the same check — and the mount disappears from their
next project or share listing. See [How share validity is decided](#how-share-validity-is-decided).

---

## Sealed Sharing (OneDrive / Dropbox / Google Drive)

### How it works

Sealing encrypts every content file of a project in place with a per-project data key
(DEK). The DEK is wrapped separately for each grantee with key material derived from their
share token, so only the holder of a share string can unwrap it. When a grantee queries,
Axon decrypts the project into a temporary cache on their machine, queries that, and wipes
it securely when the session ends (or after every query with
`security.seal_cache_ephemeral: true`).

Your own sealed projects are protected by a **master key**, created once per store and
wrapped under your passphrase with scrypt. The wrapped master is kept in the OS keyring
(DPAPI, Keychain, Secret Service) and in `<store>/AxonStore/<you>/.security/master.enc`.
**Losing the passphrase loses every sealed project — there is no recovery.**

### Prerequisites

- Both machines: `pip install "axon-rag[starter]"` (or `[sealed]`).
- A folder synced fully to disk on both machines:
  - **OneDrive:** right-click the folder → *Always keep on this device*.
  - **Google Drive:** Mirror mode, not Stream mode.
  - **Dropbox:** the folder included in Selective Sync / *Make available offline*.

### Owner

Sealing and sealed share generation need the master key **unlocked in the same process**.
`axon --project-seal`, `axon --share-generate` (sealed project) and a sealed `axon --share-revoke`
prompt for the store passphrase on the terminal. `axon --store-unlock` unlocks only the
process it runs in, so it does not help a later command. Without a terminal, seal in a REPL
session (with no `axon-api` running), or against a running `axon-api` that you unlock over REST.

```bash
axon --store-init "/path/to/OneDrive/axon"
axon --project-new research --ingest /path/to/documents
axon --passphrase-generate                 # optional: a strong Diceware passphrase
axon                                       # open the REPL for the sealed steps
```

```
axon> /store bootstrap <passphrase>          first time only; afterwards: /store unlock <passphrase>
axon> /project seal research
axon> /share generate research alice --ttl-days 30
```

`/project seal` rewrites each content file atomically (write a `.sealing` sidecar, sync,
rename). `version.json` stays plaintext so grantees can see that you re-ingested without
the key.

Through a running server instead: `POST /security/bootstrap` (once) or
`POST /security/unlock` with `{"passphrase": ...}`, then `POST /project/seal
{"project_name": "research"}` and `POST /share/generate`; the server stays unlocked until
it stops or you call `POST /security/lock`. VS Code's *Share Project* command and the
`share_project` tool go through the server the same way (`409` while it is locked).

Wait until the sync client has uploaded everything — including the share's `.wrapped` and
`.expiry` files under `research/.security/shares/` — before sending the share string.

### Grantee

```bash
axon --store-init "/path/to/OneDrive/axon"        # the same synced folder
axon --share-redeem "<share string>"
axon --project mounts/owner_research "What does the corpus say about X?"
```

Redeeming unwraps the project key and stores it according to `security.keyring_mode`
(`persistent`: the OS keyring; `session`: memory only; `never`: re-derived every time —
override per process with `--keyring-mode`). The share token itself is not kept. In the
REPL: `/share redeem <string>`, then `/project switch mounts/owner_research`.

### Envelopes: SEALED1 and SEALED2

The base64 share string decodes to a colon-separated envelope:

| Envelope | Fields | Status |
|---|---|---|
| `SEALED2:` | `SEALED2:<key_id>:<token_hex>:<owner>:<project>:<store_path>:<owner_pubkey_hex>` | Minted by every release since 0.4.0. The last field is the owner's Ed25519 signing key (derived from the master with HKDF-SHA256, `info=b"axon-share-signing-v1"`), used to verify the expiry sidecar. The pubkey is split off first, so a Windows `store_path` with `C:\` parses correctly |
| `SEALED1:` | The same without the pubkey | Legacy, pre-0.4.0. Still redeemable, but without a recorded pubkey an expiry sidecar can't be verified, so if one appears the mount fails closed. Re-issue as `SEALED2` and revoke the old key |

### Expiry

`--ttl-days` on a sealed share writes an owner-signed sidecar next to the wrap:

```
<owner-store>/<project>/.security/shares/<key_id>.wrapped    wrapped project key (~40 bytes)
<owner-store>/<project>/.security/shares/<key_id>.expiry     signed expiry (~250 bytes)
```

```json
{
  "key_id": "ssk_a1b2c3d4",
  "expires_at": "2026-06-01T17:30:00Z",
  "sig": "<base64url Ed25519 signature over b'ssk_a1b2c3d4:2026-06-01T17:30:00Z'>"
}
```

On every switch to the mount, before every query and in every listing, the grantee's Axon
reads the sidecar, verifies the signature with the pubkey recorded at redeem time
(`owner_pubkey_hex` in `mounts/<name>/mount.json`) and compares `expires_at` with the
current UTC time. A missing sidecar means no expiry. An unreadable one (cloud placeholder,
sync in flight) makes the share `unverifiable`: refused, nothing deleted. A malformed or
tampered sidecar, a `key_id` mismatch, a naive timestamp, or a time in the past makes it
`expired`, and Axon **auto-destroys the grantee's local state**:

1. the project key in the OS keyring (`axon.share.<key_id>`),
2. the file fallback `<grantee-user-dir>/.security/shares/<key_id>.dek.wrapped`,
3. the active plaintext cache for that project (`axon-sealed-*` in the temp directory),
4. the mount descriptor `mounts/<mount_name>/mount.json`, so the project disappears.

**The encrypted files on the synced folder are never touched** — deleting them would sync
the deletion back to the owner. The comparison uses the grantee's clock: a clock behind the
owner's fires late, one ahead fires early (redeem a fresh share to recover).

**Renewing a sealed share.** The sidecar is signed and can't be re-signed in place, so
`--share-extend` / `POST /share/extend` only work on plaintext shares (sealed keys return
`404`). Issue a new share with the new `--ttl-days`, send it, and once the grantee has
switched over revoke the old key (`--rotate` only if their machine is compromised).

### Revoking sealed access

**Soft revoke** — fast; the grantee's cached key survives:

```bash
axon --share-revoke ssk_abc123 --share-project research
```

REPL: `/share revoke ssk_abc123 --project research`. Deletes the share's `.wrapped`, `.kek`
and `.expiry` files, so the string can't be redeemed again. Once the deletion syncs, Axon
refuses the mount on the grantee's next switch or query and drops it from their next
listing. The key stays in the grantee's keyring, though — soft revoke relies on the
grantee running unmodified Axon. Fine for a cooperative grantee or a lost laptop.

**Hard revoke** — re-encrypts everything under a new project key:

```
axon> /store unlock <passphrase>
axon> /share revoke ssk_abc123 --project research --rotate
```

(or `POST /share/revoke {"key_id", "project", "rotate": true}` on an unlocked server).
Surviving grantees whose share has a per-share KEK
sidecar (`<key_id>.kek`) are re-wrapped automatically; the result lists
`invalidated_share_key_ids` for grantees who need a fresh share. The revoked grantee's
cached key no longer matches and their next query fails. Rotation re-encrypts every file,
so it takes about as long as sealing did, and takes effect for the grantee once the sync
client has uploaded the new files. Hard revoke is human-only: agents can soft-revoke, not
rotate.

### Security properties

| Threat | Protected? |
|---|---|
| The cloud provider reads your files | Yes — AES-256-GCM ciphertext only |
| Another collaborator on the synced folder reads your project | Yes — no key |
| Grantee keeps querying after soft revoke | Partly — refused once the wrap deletion syncs, but the key stays in their keyring; modified software could still decrypt |
| Grantee keeps querying after hard revoke | Yes — the new key is never wrapped for them |
| Plaintext on the grantee's disk during queries | Partly — only in the temp cache, wiped on exit (per query with `seal_cache_ephemeral`) |
| File sizes reveal document lengths | Optional — `security.seal_padding_bytes` adds random padding |

### Troubleshooting sealed sharing

| Error | Cause | Fix |
|---|---|---|
| `Store … is locked. Call unlock_store first.` / `SecurityError: Store is locked` | The master key isn't unlocked in this process (no terminal to prompt on) | Run the command from a terminal so it can ask for the passphrase, unlock in a REPL session, or `POST /security/unlock` on the server |
| `SecurityError: Project DEK file missing` | Not sealed yet, or not synced yet | Owner: `/project seal <name>`; grantee: wait for sync |
| `CacheCapacityError: Not enough disk space` | The temp directory needs about 1.1× the project size | Free space, or point `TMPDIR` / `TEMP` at a larger volume |
| `InvalidTag` / wrapped key won't unwrap | A hard revoke happened; the grantee's key is stale | Owner issues a new share; grantee redeems it |
| `… has been revoked by the owner (share wrap removed)` | Soft or hard revoke | Ask the owner for a new share |
| `… cannot be verified` (state `unverifiable`) | The owner's project, share directory, sealed marker or `.expiry` isn't readable yet | Wait for sync / *Always keep on this device*, retry; the mount is kept |
| `Share '<project>' (key <id>) expired at <ts>` | `expires_at` passed | New share from the owner |
| `… failed signature or format verification; treating the share as expired` | Tampered sidecar, rotated owner key, or a wrong file synced | New share from the owner; local key, cache and descriptor were auto-destroyed |
| `404` from `POST /share/extend` on an `ssk_` key | Sealed shares can't be extended | Issue a new share and revoke the old one |
| Cloud icons on project files | Files On-Demand placeholders fail mid-query | *Always keep on this device* |
| Google Drive evicts files | Stream mode | Switch to Mirror mode |

---

## Listing shares

```bash
axon --share-list            # REPL /share list · REST GET /share/list · MCP list_shares
```

`sharing` lists the shares you issued (with `revoked`), `shared` the ones you redeemed (with
their mount names). Every record carries `state` (`valid`, `revoked`, `expired`,
`unverifiable`, `invalid`) and a `reason` code from the same check the switch and query
paths use. Listing also reconciles: revoked or expired mounts are removed (REST reports
them under `removed_stale`); `unverifiable` ones are kept but refused. `GET /projects` shows
the same `state` / `reason` on each entry of `shared_mounts`.

A mounted share sees the owner's re-ingests automatically: with
`security.mount_refresh_mode: switch` (default) the grantee re-reads the owner's version
marker at most every `mount_refresh_ttl_s` (300 s) during queries; `per_query` checks before
every retrieval; `off` waits for a manual `/mount-refresh`, `axon --mount-refresh` or
`POST /mount/refresh`. While the owner's files are still syncing, queries retry and then
fail with a sync-pending error (REST `503` with `X-Axon-Mount-Sync-Pending: true`).

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

`GET /projects` (`shared_mounts`), `GET /share/list` and `list_share_mounts()` report the decision as additive `state` / `reason` fields; `is_broken` is simply "not `valid`". The owner-side view uses the same records: `POST /project/delete/{name}` is blocked only by shares that are still `valid` (plaintext or sealed) — revoked or expired shares no longer block deletion. The owner's sealed view reads the `.expiry` timestamp without verifying its signature (that needs the unlocked master key).

A corrupt `.share_manifest.json` or `.share_keys.json` is never silently rewritten: generate / revoke / extend refuse with a "refusing to overwrite" error (REST: `409`) naming the file, so revocation records can't be lost to a truncated sync.

---

## Moving a Sealed Project to a Different OS

Sealed files are the same on every platform (big-endian header, standard nonce/tag layout,
forward-slash paths in the authenticated data), and the decrypted indexes (TurboQuantDB,
LanceDB, BM25) open on any OS. What you carry across is the **master key**:

1. Copy the sealed project directory to the same relative place on the new machine, or
   point `store.base` at the shared synced folder.
2. Copy `<store>/AxonStore/<owner>/.security/master.enc` to the same relative path. Axon
   writes it next to the keyring entry on every platform.
3. Unlock with the same passphrase — `/store unlock <passphrase>` in the REPL (or
   `POST /security/unlock`). With no keyring entry on the new machine, Axon falls back to
   `master.enc` by itself.
4. Switch to the project and query as usual.

`master.enc` is protected by your passphrase through scrypt (N=2¹⁵). Without the
passphrase it is useless, but treat it like a password-manager export.

---

## Filesystem Compatibility Matrix

This matrix is for **plaintext** sharing. Sealed sharing works on any of these, because
only ciphertext reaches the filesystem.

| Filesystem | Verdict | Notes |
|---|---|---|
| **Local disk** (NTFS / ext4 / APFS / ZFS) | Safe | One owner writing, grantees on the same machine reading. |
| **On-premises SMB3, Windows Server 2019+** | Safe, with caveats | Grantees must be Windows-native (not WSL). SMB3 leases keep readers coherent. Keep SQLite-backed state off the share. |
| **DFS Namespace (DFS-N, without DFS-R)** | Thin alias | A referral layer over one SMB server; inherits that share's behaviour. |
| **Azure Files (SMB 3.1.1)** | Usable for reads | Continuous-availability retries hang clients for minutes during drops. Keep `.dynamic_graph.db` on local disk. |
| **OneDrive** (Personal / Business / SharePoint) | Unsafe for plaintext | Files On-Demand placeholders hang memory-mapped reads; `-wal` / `-shm` sidecars sync out of order and corrupt SQLite; conflict copies are ignored. Use sealed. |
| **Dropbox** | Unsafe for plaintext | Same sidecar reordering; conflicted copies ignored. Use sealed. |
| **Google Drive for Desktop** | Unsafe for plaintext | Same SQLite corruption; `.tmp.drivedownload` clutter; Stream mode evicts files mid-query. Use sealed (Mirror mode). |
| **DFS Replication (DFS-R)** | Unsafe | `ConflictAndDeleted` silently eats "losing" index files; 15-minute minimum replication. |
| **WebDAV redirector** | Unsafe | 50 MB default file cap; stale directory caches. |
| **WSL on a Windows mount** (`/mnt/c/...`) | Unsafe for the owner | WSL1 `fcntl(F_SETLK)` is broken; WSL2's CIFS emulation is unpredictable. Symptom: `attempt to write a readonly database`. |

Axon's own mitigations: the `dynamic_graph` backend uses SQLite's `DELETE` journal mode
(no `-wal` / `-shm`), and grantees read a JSON snapshot (`.dynamic_graph.snapshot.json`)
instead of opening the owner's database. `axon --config-validate` warns when the store is
on a cloud-sync or network path.

| Filesystem | Recommended vector store | Why |
|---|---|---|
| Local disk | TurboQuantDB (default) | Single-file memory map; best recall per byte |
| SMB3 / DFS-N | TurboQuantDB or LanceDB | TurboQuantDB for small and medium corpora; LanceDB's immutable fragments replicate cleanly if the owner compacts on a schedule |
| Azure Files | TurboQuantDB | Keep SQLite-backed state off the share |

Don't use Chroma on any shared, network or cloud-sync path; Chroma is not supported for
sealed projects either.

---

## Security Considerations

### Windows users with highly sensitive data

The decrypted cache is overwritten with random bytes before deletion, and on Windows Axon
calls `FlushFileBuffers` after each wipe. NTFS copy-on-write and SSD TRIM / wear-levelling
can still leave plaintext in freed sectors, so treat the wipe as best-effort. Axon logs
`SealedCache: Windows NTFS secure-delete is best-effort` at INFO the first time it wipes a
cache directory.

For compliance or classified use:

- **Enable BitLocker** on the drive holding `%TEMP%` (where the cache lives by default) —
  freed sectors stay encrypted at rest.
- **Move the cache to an encrypted volume or RAM disk.** The cache is created in Python's
  temporary directory, so set `TEMP` / `TMP` (Windows) or `TMPDIR` (Linux/macOS) for the
  Axon process to a BitLocker drive, a VeraCrypt container or a RAM disk:
  ```
  set TEMP=E:\secure-tmp
  set TMP=E:\secure-tmp
  ```
- **Use `security.seal_cache_ephemeral: true`** (or `--seal-cache-ephemeral`) to keep
  plaintext on disk only while a query runs, at about a second of extra latency per query.

### Passphrase strength

`--store-bootstrap` and `--store-change-passphrase` (and their REPL / REST forms) require
at least 8 characters. That is a floor: the master is wrapped with scrypt (N=2¹⁵, r=8,
p=1), and a 16+ character random passphrase or a 6-word Diceware phrase
(`axon --passphrase-generate`, ≈ 77 bits) makes offline guessing impractical. Unlock does
not enforce the minimum, so a wrong passphrase is reported as wrong rather than too short.

---

## See Also

- [Reference](REFERENCE.md) — every share, seal and store command on each surface
  ([CLI](REFERENCE.md#66-store-sealing-and-sharing), [REPL](REFERENCE.md#75-store-and-sharing),
  [REST](REFERENCE.md#86-sharing-bodies), [MCP](REFERENCE.md#93-tools))
- [Troubleshooting](TROUBLESHOOTING.md) — broader error patterns
- [Sealed sharing design](architecture/SEALED_SHARING_DESIGN.md) — key hierarchy and threat model
- [`scripts/qa/SEALED_SHARE_SMOKE.md`](../scripts/qa/SEALED_SHARE_SMOKE.md) — a two-machine manual test recipe
