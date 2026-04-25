# Real-sync E2E tests for sealed shares (Phase 7 Layer 2)

## What this layer covers

These tests bring up a **Nextcloud server in Docker** and exercise the
sealed-share upload / download / two-writer race / revoke flow over its
WebDAV interface. No real OneDrive / Microsoft account is needed.

Why Nextcloud as a OneDrive substitute: at the file-level abstraction
Axon cares about, OneDrive's sync semantics are isomorphic to
Nextcloud's WebDAV — same eventual consistency, same conflict-file
naming convention (`*-conflict-*`), same ETag-based change detection,
same lock contention. Axon treats the synced folder as **opaque
ciphertext** anyway, so we don't need the OneDrive *client* — only the
sync semantics.

## Coverage matrix

| Failure mode                                | Layer 1 (`tests/sync/`) | **Layer 2 (this dir)** | Manual smoke |
| ------------------------------------------- | :---------------------: | :--------------------: | :----------: |
| Partial-byte visibility during upload       |            ✅            |                        |       ✅      |
| Conflict-file naming (`*-conflict-*`)       |          chaos          |        **real**        |       ✅      |
| `.tmp.drivedownload` / temp-file debris     |            ✅            |                        |       ✅      |
| ETag-based change detection                 |                         |        **real**        |       ✅      |
| Two-writer race producing 412 Precondition  |                         |        **real**        |       ✅      |
| Eventual-consistency settle window          |                         |        **real**        |       ✅      |
| Soft-revoke via wrap delete                 |          mocked         |        **real**        |       ✅      |
| OneDrive Files-On-Demand placeholder        |                         |                        |     **✅**    |
| Windows Explorer / Defender file lock       |                         |                        |     **✅**    |
| Microsoft Graph throttling at scale         |                         |                        |     **✅**    |

The bottom three rows are fundamentally Windows-kernel / Microsoft-
specific and stay covered by `docs/SHARE_MOUNT_SEALED_SMOKE.md` (run
quarterly + before each release).

## Prerequisites

- **Docker** (Docker Desktop on Windows / macOS, or `docker-ce` on Linux).
  The fixture probes the daemon with `docker info` at suite setup; if
  the daemon isn't reachable, the suite **skips cleanly** rather than
  failing. Same for `docker compose up` errors (image pull, network).

- **Python deps:** the `[sealed-test]` extra adds `requests`. Install with:

  ```bash
  pip install -e ".[sealed-test]"
  ```

## Run

```bash
# Run just the Layer 2 suite
python -m pytest tests/e2e_sync/ --no-cov -v

# Skip on demand (e.g. offline)
SKIP_DOCKER=1 python -m pytest tests/e2e_sync/ --no-cov   # if you've taught the conftest about it
```

First run pulls the `nextcloud:30-apache` image (~ 200 MB) — takes a
minute. Subsequent runs reuse the cached image and finish in ~30–45 s.

## When the tests run

These tests are **gated by Docker availability**. If your dev machine
doesn't have Docker, all 5 tests skip with a clear reason (no error).
You're not blocked from committing.

In CI: the suite runs as part of the standard pytest invocation in
`.github/workflows/ci.yml`. The GitHub-hosted Ubuntu runners have
Docker available, so the Layer 2 suite executes there automatically;
the macOS / Windows matrix runners skip because Docker Desktop isn't
installed on those images. A dedicated path-filtered workflow that
runs Layer 2 ONLY when sealed-share code changes is tracked as a
follow-up — for now Layer 2 runs on every CI invocation that hits
an Ubuntu runner.

In pre-commit: **not added by default** — Docker is a heavy dep to
require for every commit. Devs working on sealed-share code can
opt in by adding the suite to their personal `.pre-commit-config.yaml`.

## What the tests prove

1. **`TestSealedShareWebDavRoundTrip`** — wrap files survive the
   server round trip; ETag mismatch detection works; soft-revoke
   produces 404s on the grantee side.
2. **`TestSealedShareSyncListing`** — `list_sealed_share_key_ids`
   filters out sync-engine debris (conflict copies, temp downloads)
   that would otherwise pollute `hard_revoke`'s invalidation list.
3. **`TestSealedShareEndToEndViaNextcloud`** — full flow: owner
   seals + uploads via WebDAV; grantee downloads via WebDAV; grantee's
   `redeem_sealed_share` produces a DEK identical to the owner's.

## Why no real OneDrive option here

Three independent blockers:

1. **Microsoft 365 Developer Program ToS** explicitly forbids using
   sandbox tenants as a CI fixture for third-party tools.
2. **Microsoft Graph throttling** (3,500–8,000 ResourceUnits/10s per
   app+tenant) → a CI matrix would 429 immediately.
3. **OAuth in headless CI** → device flow needs a browser; refresh
   tokens stored in CI secrets get revoked by Conditional Access
   rotations.

A separate (currently unimplemented) **opt-in local pre-commit hook**
COULD use the dev's already-signed-in OneDrive client + Microsoft
Graph SDK + DPAPI-cached refresh token. Reference: `abraunegg/onedrive`
runs that pattern in its CI. We chose Nextcloud-in-Docker as the
default because it covers ~95% of what matters without per-dev setup.
See `docs/SHARE_MOUNT_SEALED_SMOKE.md` for the full reasoning.
