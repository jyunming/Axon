# Sealed-Share Smoke Test — Manual Recipe (Phase 7)

> **Status:** Manual checklist run by hand on two real machines before tagging a release that touches the sealed-store stack. Pairs with the automated chaos suite at `tests/sync/` (Phase 7 Layer 1) which catches partial-byte / locked-file / conflict-copy regressions on every PR but cannot exercise the real cloud-sync engine.

This is a stripped-down version of `docs/SHARE_MOUNT_SMOKE.md` (the legacy share-mount smoke recipe from PR #55) extended with sealed-store steps from Phases 2–6. Run it end-to-end on **two physical machines** sharing a OneDrive / Dropbox / SMB folder.

---

## What this recipe covers (and what it doesn't)

Three-tier testing strategy:

| Failure mode                                           | Layer 1 — FS chaos (`tests/sync/`) | Layer 2 — Nextcloud-Docker (`tests/e2e_sync/`) | Manual smoke (this recipe) | Real cloud only |
| ------------------------------------------------------ | :--------------------------------: | :--------------------------------------------: | :------------------------: | :-------------: |
| Partial-byte visibility during upload                  |                  ✅                |                                                |              ✅            |                 |
| `.tmp.drivedownload` / OneDrive temp-file debris       |                  ✅                |                                                |              ✅            |                 |
| OneDrive conflict-copy filenames in shares dir         |                  ✅                |                                                |              ✅            |                 |
| Sync-engine exclusive lock on a wrap file              |                  ✅                |                                                |              ✅            |                 |
| ETag-based change detection                            |                                    |                       ✅                       |              ✅            |                 |
| True two-writer race (412 Precondition Failed)         |                                    |                       ✅                       |              ✅            |                 |
| Eventual-consistency settle window (variable lag)      |                                    |                       ✅                       |              ✅            |                 |
| Soft-revoke → grantee 404                              |                                    |                       ✅                       |              ✅            |                 |
| **Microsoft Graph throttling / 429 patterns**          |                                    |                                                |                            |        ✅       |
| **Files-On-Demand placeholder semantics (Windows)**    |                                    |                                                |              ✅            |                 |
| **Windows Explorer preview-pane handle lock**          |                                    |                                                |              ✅            |                 |
| **Real OAuth refresh-token expiry mid-sync**           |                                    |                                                |                            |        ✅       |

- **Layer 1** runs every PR (no extra deps, ~5 s).
- **Layer 2** runs whenever the standard pytest workflow hits a CI runner that has Docker (currently the Ubuntu matrix cells; macOS/Windows skip cleanly). A path-filtered "only on sealed-share changes" workflow is tracked as future work.
- **This manual recipe** runs pre-release on a real two-machine OneDrive setup (covers Windows-kernel + UI specifics that no automated layer can reach).

**Why no automated real-OneDrive layer:** Microsoft 365 Developer Program ToS forbids using sandbox tenants as CI fixtures, Microsoft Graph throttling caps a single app+tenant at 3,500–8,000 ResourceUnits per 10s (a CI matrix would 429 immediately), and OAuth in headless CI is structurally brittle (refresh tokens get rotated by Conditional Access). Local-on-dev real-OneDrive is technically possible (`abraunegg/onedrive` does this in its own CI) — see `tests/e2e_sync/README.md` for the opt-in recipe. We chose Nextcloud-in-Docker as the default because it covers ~95% of what matters without per-dev OneDrive account setup.

---

## Setup

You need:

- **Two machines** (call them `OWNER` and `GRANTEE`).
- A **shared sync folder** mounted on both (OneDrive Personal / Business, Dropbox, Google Drive Desktop in Mirror mode, or an SMB share). Note the local path on each machine — they may differ.
- Axon installed with the `[sealed]` extra on both machines: `pip install axon-rag[sealed]`.

Setup convention used below:

```
OWNER  : C:\Users\alice\OneDrive\AxonStore\alice\
GRANTEE: C:\Users\bob\OneDrive\AxonStore\alice\
```

(Same `alice` username on both — the sync engine mirrors the folder.)

---

## Recipe

### Step 1 — Owner: bootstrap + seal

```bash
# OWNER machine
axon --store-init "C:\Users\alice\OneDrive"
axon --store-bootstrap "owner-passphrase-PLEASE-USE-A-REAL-ONE"
axon --project-new research
axon --project research --ingest "C:\path\to\some\docs"
axon --project-seal research
axon --store-status   # confirm: initialized=True, unlocked=True
```

**Expect:** `Project 'research': sealed (N files)`. Inspect on disk:

```powershell
ls "C:\Users\alice\OneDrive\AxonStore\alice\research\.security\"
# → dek.wrapped, .sealed, shares\ (empty)
ls "C:\Users\alice\OneDrive\AxonStore\alice\research\bm25_index\"
# → .bm25_log.jsonl ← should be AXSL ciphertext now
```

Run `head -c 4` on a content file:

```powershell
[System.Text.Encoding]::ASCII.GetString((Get-Content -AsByteStream -TotalCount 4 "...\research\meta.json"))
# → AXSL
```

✅ Owner has sealed at-rest.

### Step 2 — Wait for sync

Watch the OneDrive system-tray icon until everything is uploaded. **Do not skip this step** — Step 3 fails confusingly if the sync hasn't finished.

### Step 3 — Owner: generate a sealed share

```bash
# OWNER machine
axon --project-seal research   # (no-op — already sealed)
# Generate via REST since the sealed-share generate is auto-detected:
curl -X POST http://localhost:8000/share/generate \
     -H "Content-Type: application/json" \
     -d '{"project":"research","grantee":"bob"}'
```

**Expect** the JSON response carries `share_string` starting with a base64 envelope whose decoded form starts with `SEALED1:`. Save the `share_string` and the `key_id` (the `ssk_xxx` value).

✅ Owner has a sealed share envelope to transmit.

### Step 4 — Wait for sync (again)

The owner just wrote `<project>/.security/shares/<key_id>.wrapped`. Wait for that 40-byte file to upload.

### Step 5 — Grantee: redeem

Transmit the `share_string` from Step 3 to the GRANTEE out-of-band (Slack, paper note — anything except the synced folder itself, which a passive observer could read).

```bash
# GRANTEE machine
axon --share-redeem "<paste the share_string here>"
```

**Expect:** `Share redeemed!` with the mount printed. Inspect:

```powershell
cat "C:\Users\bob\OneDrive\AxonStore\bob\mounts\alice_research\mount.json"
# → mount_type: "sealed", share_key_id: "ssk_xxx"
```

✅ Grantee has a sealed mount descriptor + the unwrapped DEK in their OS keyring.

### Step 6 — Grantee: query the sealed mount

```bash
# GRANTEE machine
axon --project mounts/alice_research "what does the corpus say about X?"
```

**Expect:** A real answer drawn from the owner's sealed corpus. Behind the scenes:

1. `switch_project mounts/alice_research` reads the mount descriptor, sees `mount_type=sealed`.
2. Fetches the DEK from the OS keyring (`axon.share.ssk_xxx`).
3. `materialize_for_read` decrypts the project into an ephemeral cache under `%LOCALAPPDATA%\Temp\axon-sealed-*\`.
4. The standard query path runs against the cache.
5. On process exit (or next `switch_project`), the cache is securely wiped.

Verify cache wipe:

```powershell
ls "$env:LOCALAPPDATA\Temp\axon-sealed-*"
# → empty after axon exits
```

✅ End-to-end sealed-share works across two machines via the sync engine.

### Step 7 — Owner: soft revoke

```bash
# OWNER machine
curl -X POST http://localhost:8000/share/revoke \
     -H "Content-Type: application/json" \
     -d '{"key_id":"ssk_xxx","project":"research"}'
```

**Expect:** `{"status":"soft_revoked", "wrap_deleted": true}`. The wrap file disappears from disk → sync deletes it on the grantee side too.

### Step 8 — Wait for sync

### Step 9 — Grantee: confirm soft-revoke effects

```bash
# GRANTEE machine — fresh redeem of the SAME share_string fails
axon --share-redeem "<the same share_string>"
# → Sealed-share wrap file missing at .../shares/ssk_xxx.wrapped — owner has revoked.

# But the existing mount still works because the DEK is cached in the keyring
axon --project mounts/alice_research "another question?"
# → STILL ANSWERS  ← documented soft-revoke trade-off (Phase 4 design)
```

✅ Soft revoke matches docs: cached DEK still works; fresh redeem blocked.

### Step 10 — Owner: hard revoke (rotate)

```bash
# OWNER — hard revoke rotates the project DEK + invalidates ALL shares
curl -X POST http://localhost:8000/share/revoke \
     -H "Content-Type: application/json" \
     -d '{"key_id":"ssk_xxx","project":"research","rotate":true}'
```

**Expect:** `{"status":"hard_revoked", "files_resealed": N, "invalidated_share_key_ids": [...]}`. Every content file under `research/` is now encrypted under the new DEK; `dek.wrapped` is updated; every share wrap is gone.

### Step 11 — Wait for sync

This step uploads N re-encrypted content files plus the new `dek.wrapped`. Larger projects = longer wait. Watch the tray icon.

### Step 12 — Grantee: confirm hard-revoke effects

```bash
# GRANTEE — the cached DEK no longer matches the new ciphertext.
axon --project mounts/alice_research "still answering?"
# → Materialise fails with InvalidTag — the cached DEK is stale.
```

To recover, the OWNER re-issues a NEW share with a fresh `key_id`, the GRANTEE redeems it, and queries work again.

✅ Hard revoke matches docs: cached DEK is invalidated.

---

## Failure-mode probes (run only if a regression is suspected)

**Mid-sync redeem (the partial-byte gap):**

1. OWNER generates a share but kills the network before sync uploads the wrap file.
2. GRANTEE attempts redeem.
3. **Expect:** "Sealed-share wrap file missing at ... The owner has revoked OR the file has not yet synced. Wait and retry."

**Conflict copy (two-machine race):**

1. OWNER generates `ssk_a`; wait for sync.
2. Disconnect OWNER's network. OWNER hard-revokes (rotates).
3. GRANTEE doesn't see the rotation yet.
4. Re-enable OWNER's network. Wait for sync — OneDrive will create a conflict copy of `dek.wrapped` if there was a clobber attempt.
5. **Expect:** GRANTEE's queries still fail with InvalidTag (rotation took effect on the canonical file). Conflict copy is a `.wrapped`-suffixed file that `list_sealed_share_key_ids` ignores (filtered out by the strict key_id pattern).

**Sync-engine lock:**

1. While GRANTEE has a `mounts/alice_research` query running, OWNER tries to hard-revoke.
2. **Expect:** Hard revoke succeeds atomically per file; staged sidecars (`dek.wrapped.rotating` + `.sealing.rotation`) survive any temporary lock collisions and the next attempt resumes.

---

## When this recipe must run

- Before tagging any release that touches `axon/security/` (Phases 2–6).
- After cloud-sync app updates (OneDrive feature changes can break Files-On-Demand assumptions).
- When the user or the issue tracker reports a "works locally, fails on OneDrive" symptom.

Otherwise rely on the chaos suite (`tests/sync/`) for per-PR coverage.
