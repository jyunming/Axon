# Share Mount Guide

Axon's share-mount feature lets one user (the **owner**) expose a project
read-only to other users (the **grantees**) by sharing the on-disk project
directory through a coherent filesystem. This doc tells you which
filesystems Axon supports as the shared layer, which ones *will* corrupt
your data, and why.

> **TL;DR.** Share via on-prem SMB3 from Windows-native machines, or
> via a local/LAN copy protocol. **Do not** share via OneDrive, Dropbox,
> Google Drive, WebDAV, or DFS-R.
>
> A future server-mediated mount design (no shared filesystem at all,
> bytes never copied to grantees) is sketched in
> [SHARE_MOUNT_REMOTE.md](SHARE_MOUNT_REMOTE.md). For verifying the
> *current* file-sharing model against real OneDrive, see
> [SHARE_MOUNT_SMOKE.md](SHARE_MOUNT_SMOKE.md).

See also: `docs/AXON_STORE.md` (how the descriptor-backed mount model
works) and `docs/TROUBLESHOOTING.md` (what to do when something has
already gone wrong).

## How sharing works today

An Axon mount is a JSON descriptor (`mount.json`) stored under
`~/.axon/AxonStore/{you}/mounts/{name}/`. It points at the **owner's**
`target_project_dir`. When you activate the mount, Axon opens the
owner's `vector_store_data/`, `bm25_index/`, `meta.json`, and graph
state **directly** — no bytes are copied by Axon itself.

This means **multi-machine** sharing works only if the owner's
`target_project_dir` is visible to the grantee's machine on a filesystem
that behaves like a real local disk — specifically, with atomic file
rename and reliable advisory locking. Whatever puts the bytes in front
of the grantee (SMB, cloud sync, etc.) is responsible for that
guarantee. Axon does not sync anything itself.

## Supported filesystems

| Filesystem | Verdict | Notes |
|---|---|---|
| **Local disk** (NTFS / ext4 / APFS / ZFS) | ✅ Safe | The canonical path. Single-writer owner, many-reader grantees on the same machine works out of the box. |
| **On-prem SMB3 on Windows Server 2019+** | ✅ Safe, with caveats | Grantees must be **Windows-native** (not WSL). SMB3 leases give the reader a coherent view. Avoid SQLite-backed components on the share even so. |
| **DFS Namespace (DFS-N, without DFS-R)** | ⚠️ Thin alias | Pure referral layer over a single SMB server. Inherits whatever that SMB share gives you. |
| **Azure Files (SMB3.1.1)** | ⚠️ Usable for reads | Continuous-Availability retry window hangs clients for minutes during drops. Keep `.governance.db` on local disk. |

## Unsupported filesystems — do not use

| Filesystem | Verdict | Why |
|---|---|---|
| **OneDrive** (Personal / Business / SharePoint) | ❌ Unsafe | Files On-Demand placeholders hang `mmap`; `-wal`/`-shm` sidecars sync out of order and corrupt SQLite; conflict copies named `file-CONFLICT-HOSTNAME.ext` are silently ignored by Axon. Documented [git corruption pattern](https://techcommunity.microsoft.com/discussions/onedriveforbusiness/onedrive-is-corrupting-my-git-repositories/3898283) applies 1:1. |
| **Dropbox** (Personal / Business) | ❌ Unsafe | Same SQLite sidecar reorder; `conflicted copy` files silently ignored; JetBrains-class "constant file operations" collide with active index writes. |
| **Google Drive for Desktop** (Mirror or Stream) | ❌ Unsafe | Same SQLite corruption; `.tmp.drivedownload` clutter; Stream-mode cache evicts mid-query. |
| **DFS Replication (DFS-R)** | ❌ Unsafe | Microsoft explicitly warns DFS-R is [not for multi-writer scenarios](https://learn.microsoft.com/en-us/windows-server/troubleshoot/understanding-the-lack-of-distributed-file-locking-in-dfsr); `ConflictAndDeleted` silently eats "losing" index files; 15-min minimum replication interval. |
| **WebDAV redirector** | ❌ Unsafe | 50 MB default file cap, 4 GB hard cap (below typical vector stores); `WebClient` service deprecated since Nov 2023; directory caches serve stale reads. |
| **WSL + Windows mount** (`/mnt/c/...`) | ❌ Unsafe for the owner | WSL1's `fcntl(F_SETLK)` is broken; WSL2's cifs emulation is unpredictable. Symptom: `"attempt to write a readonly database (code: 8)"`. You'll see a loud error from Axon before any data is touched. |

If you set up Axon on one of these, the config validator will emit a
`warn` issue (`axon --config-validate`, REST `GET /config/validate`)
and `OpenVectorStore._raise_unsupported_path` will raise a `RuntimeError`
with the same diagnosis when you try to use Chroma.

## Recommended backend per supported filesystem

| Filesystem | Recommended backend | Rationale |
|---|---|---|
| Local disk | **TurboQuantDB** (default) | Single-binary mmap, best recall/size tradeoff. |
| SMB3 / DFS-N | TurboQuantDB or LanceDB | TurboQuantDB for small/medium corpora; LanceDB's immutable fragments replicate cleanly when the owner compacts on a schedule. |
| Azure Files | TurboQuantDB | Keep `.governance.db` off the share (relocation is automatic on v0.2.2+). |

Do **not** use Chroma on any shared / network / cloud-sync path. Axon's
default (TurboQuantDB) does not use SQLite and is the right choice for
shared scenarios.

## What Axon already does to protect you

As of v0.2.2 (branch `fix/share-mount-sqlite-wal-safety`):

- The governance audit DB (`.governance.db`) runs in journal mode
  `DELETE`, not `WAL`, so no `-wal`/`-shm` sidecars exist to be
  reordered by a sync client.
- The Dynamic Graph backend (`.dynamic_graph.db`) runs in `DELETE`
  mode and is automatically relocated to `~/.axon/graphs/{project_id}/`
  when `brain.config.bm25_path` is detected as cloud-sync / network /
  WSL-mount. An export is written to `bm25_index/.dynamic_graph.snapshot.json`
  after every ingest so grantees can still read the graph.
- Grantees of a shared project never open the owner's
  `.dynamic_graph.db` — they load the JSON snapshot into an
  in-memory SQLite.
- `AxonConfig.validate()` flags `store.base`, `vector_store.path`, and
  `bm25.path` when they land on an unsafe filesystem, with a pointer
  to this doc.
- Chroma's `_init_store()` does a preflight path check and raises a
  loud `RuntimeError` before attempting to open a DB on an unsafe
  filesystem.

Still pending (see open issues `#53`, `#54`):

- **Version marker for staleness detection** (`#53`) — grantees
  currently have no way to tell when the owner has re-ingested. A
  future `version.json` + refresh hook will close this gap.
- **Dual-revocation** (`#54`) — revoking an Axon share-key does not
  currently invalidate a grantee's cached `vector_store_data/`. A
  determined grantee with a hacked client could still read the
  vectors until the underlying filesystem ACL is also revoked.

Until those land, the recommendation is: keep share-mount usage on
local disk or on-prem SMB3, and treat OneDrive / Dropbox / Google Drive
as backup paths only (ship a snapshot tarball to the cloud folder on a
schedule rather than running Axon live against it).

## Troubleshooting

If you see `"attempt to write a readonly database (code: 8)"` — that's
the WSL + Windows-mount case. Move your Axon state onto the Linux
filesystem (`export AXON_STORE_BASE=~/.axon`) or switch to
TurboQuantDB.

If you see `"database disk image is malformed"` or silent wrong answers
after working inside `OneDrive/` — the `-wal`/`-shm` sidecars got
reordered by the sync client. The v0.2.2 DELETE-mode fix eliminates the
cause; if you still see it, rebuild the project on local disk and
inspect `axon --config-validate` for the warning.

If grantees see stale results after the owner ingested — this is the
#53 gap. Switch to a fresh mount (grantee `/project switch mounts/<name>`
re-reads everything) as a workaround until the refresh hook lands.

See `docs/TROUBLESHOOTING.md` for broader categories.
