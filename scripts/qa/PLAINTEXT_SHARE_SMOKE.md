# Manual smoke test: share mount under OneDrive

> **Purpose.** Verify the share-mount fixes from `fix/share-mount-sqlite-wal-safety`
> (PR pending) actually work against a **real OneDrive** sync — the CI
> tests in `tests/test_share_mount_sync_reorder.py` simulate the
> mechanism on local tmpdirs but cannot exercise an actual cloud-sync
> client. **Run this once before tagging any release that touches
> `axon/paths.py`, `axon/version_marker.py`, `axon/shares.py`, or the
> share-mount surfaces.**
>
> Estimated time: **45–60 minutes** including OneDrive sync waits.

## What's being verified

| Fix | Expected behaviour |
|---|---|
| WAL → DELETE journal mode (#51) | No `*-wal` / `*-shm` files appear in the OneDrive folder |
| Cloud-sync path classifier (#51) | `axon --config-validate` warns about the OneDrive path |
| Chroma preflight (#52) | Switching to Chroma at the OneDrive path raises a loud `RuntimeError` (no silent corruption) |
| Owner DB relocation (#51) | `.dynamic_graph.db` lives under `~/.axon/graphs/<id>/`, not in OneDrive |
| Snapshot for grantees (#51) | `bm25_index/.dynamic_graph.snapshot.json` appears after ingest and syncs |
| `version.json` marker (#53) | Marker is written last, syncs visibly, includes correct `seq` |
| `refresh_mount` (#53) | Grantee picks up owner re-ingests via `/project refresh` or REST `POST /mount/refresh` |
| `MountSyncPendingError` (#53) | Grantee sees a friendly "still replicating" message, NOT a torn read, when artifacts are mid-sync |
| Share-key TTL (#54) | Past-expiry shares are rejected at redeem AND on grantee's next query |
| `extend_share_key` (#54) | Owner can renew, grantee continues to access |

## Prerequisites

- **Two Windows machines** (or one Windows + one macOS; OneDrive
  client must be installed and signed in on both).
- **Same OneDrive account** signed in on both — a personal
  OneDrive account works; OneDrive for Business also fine.
- **Python 3.11+** + Axon installed on both: `pip install axon-rag`
  or run from a checkout of the branch under test.
- **Ollama** running on the owner box with at least one small model
  (e.g. `ollama pull llama3.2:1b`) — only the owner needs an LLM for
  ingest; the grantee just queries.
- **A small test corpus** ready on the owner box — a handful of
  `.md` / `.txt` files is enough. The repo's `docs/` directory works.

Conventions used below:
- **Box A** = owner machine. Username `alice` in examples.
- **Box B** = grantee machine. Username `bob` in examples.
- Substitute your actual OS username in commands (`getpass.getuser()`).

## Step 0 — set up OneDrive paths

On both boxes:

1. Confirm OneDrive is running and synced (white cloud icon in tray).
2. Find the OneDrive folder. Default Windows path is
   `C:\Users\<username>\OneDrive` (Personal) or
   `C:\Users\<username>\OneDrive - <Tenant>` (Business).
3. Create a fresh subfolder for this test:
   `<OneDrive root>\axon-smoke\` on **Box A only**. It will sync to
   Box B automatically — wait for the green checkmark on Box B before
   continuing.

## Step 1 — point Axon at the OneDrive folder (deliberately)

We're forcing the bad configuration the fixes are meant to mitigate.

**Box A (owner):**

```powershell
# Set the AxonStore base under OneDrive — exactly the configuration
# that would have silently corrupted SQLite before this branch.
$env:AXON_STORE_BASE = "$env:OneDrive\axon-smoke"

# Create a project and ingest a corpus.
axon --config-validate
# EXPECT: a "warn" issue mentioning "store.base" + "cloud-sync" +
# the OneDrive path. If you don't see this warning, the path
# classifier is broken — STOP and investigate.

axon
> /project new research
> /ingest C:\path\to\test\corpus
> /q What is in this corpus?
# EXPECT: a normal answer based on the ingested docs.
> /exit
```

**Verify on Box A — no SQLite WAL sidecars:**

```powershell
# These files MUST NOT exist if the WAL→DELETE fix is working.
Get-ChildItem "$env:OneDrive\axon-smoke\AxonStore\$env:USERNAME\research\bm25_index\" `
    -Filter "*.db-wal", "*.db-shm" -ErrorAction SilentlyContinue
# EXPECT: empty result. Any file listed = WAL fix regressed.

# The dynamic-graph DB should NOT live under OneDrive — it should have
# been relocated to ~/.axon/graphs/<project_id>/.
Get-ChildItem "$env:OneDrive\axon-smoke\AxonStore\$env:USERNAME\research\bm25_index\.dynamic_graph.db" `
    -ErrorAction SilentlyContinue
# EXPECT: empty (file does not exist on the synced path).

# But a snapshot SHOULD exist for grantees to read.
Test-Path "$env:OneDrive\axon-smoke\AxonStore\$env:USERNAME\research\bm25_index\.dynamic_graph.snapshot.json"
# EXPECT: True.

# The version marker should also exist, written LAST after ingest.
Test-Path "$env:OneDrive\axon-smoke\AxonStore\$env:USERNAME\research\version.json"
# EXPECT: True.

# And the relocated DB IS local under the owner's home dir.
Get-ChildItem "$env:USERPROFILE\.axon\graphs\" -Recurse -Filter ".dynamic_graph.db"
# EXPECT: at least one match.
```

If any check fails → **stop, file an issue, do not ship this release.**

## Step 2 — Chroma preflight (#52)

```powershell
# Force the Chroma backend at the OneDrive path.
$env:AXON_STORE_BASE = "$env:OneDrive\axon-smoke"
axon --config "vector_store.provider=chroma"
> /project new chroma-test
> /ingest C:\path\to\test\corpus
# EXPECT: loud RuntimeError BEFORE any files are written. The error
# message must mention OneDrive, the path, and link to docs/SHARING.md.
# If ingest silently succeeds and a chroma.sqlite3 appears in the
# OneDrive folder → preflight regressed.
```

## Step 3 — share + sync to Box B

**Box A (owner):**

```powershell
axon
> /share generate research bob --ttl-days 30
# EXPECT output includes:
#   Key ID:       sk_xxxxxxxx
#   Expires at:   2026-MM-DD...  (~30 days from now)
#   Share string: <long base64>
> /exit
```

Copy the share string to Box B (Slack / clipboard / paste-bin).

**Wait** for OneDrive to fully sync `axon-smoke/` to Box B. This may
take 30 seconds to a few minutes for a small project, longer for a
large one. Watch the OneDrive icon — it should settle on the green
checkmark.

**Box B (grantee):**

```powershell
$env:AXON_STORE_BASE = "$env:OneDrive\axon-smoke"
axon
> /share redeem <paste share string>
# EXPECT: "Share redeemed!" with the mount name.
> /project switch mounts/alice_research
# EXPECT: log line "Mounted 'mounts/alice_research' at owner version
# seq=1 (generated_at=...)"  — the version marker was read.
> /q What is in this corpus?
# EXPECT: same answer Box A got — proves grantee reads owner's data
# correctly through the synced files.
> /exit
```

If the query returns "no results" or an error → grantee read path
broken; check `.dynamic_graph.snapshot.json` was synced, check
`bm25_index/` files arrived intact.

## Step 4 — refresh after owner re-ingest (#53)

**Box A:** add new content and re-ingest:

```powershell
axon
> /project switch research
> /ingest C:\path\to\different\corpus
> /exit
```

**Wait** for OneDrive to sync the changed files to Box B. You should
see `version.json` get a newer timestamp on Box B.

**Box B (mount still active from Step 3):**

```powershell
axon
> /project switch mounts/alice_research
> /q something only in the new corpus
# EXPECT: stale answer (default mount_refresh_mode = "switch", so the
# grantee is on the seq it cached at switch time). This is correct
# behaviour, not a bug.

> /project refresh
# EXPECT: "Refreshed mount: now at owner seq=2 (generated_at=...)"
> /q something only in the new corpus
# EXPECT: correct answer using the new content.
> /exit
```

## Step 5 — mid-sync race protection

**Box A:**

```powershell
axon
> /project switch research
> /ingest C:\path\to\third\corpus
> /exit
```

**Immediately on Box B (before sync completes):**

```powershell
# Pause OneDrive on Box B to freeze the partial state.
# Right-click OneDrive tray icon → Pause syncing → 2 hours.

axon
> /project switch mounts/alice_research
> /project refresh
# Now resume OneDrive on Box B and IMMEDIATELY pause again after the
# version.json is observed but before the larger files finish.
```

This is the hardest scenario to reproduce manually because OneDrive's
sync ordering is opaque. If you can hit the window:

```
EXPECT: a clear "Owner has advanced but index files are still
replicating" message, NOT a stale or corrupt answer.
```

If you can't reproduce the race: that's fine — `tests/test_share_mount_sync_reorder.py`
covers this case mechanically. Note in the smoke report that you
were unable to provoke the race naturally.

## Step 6 — TTL enforcement (#54)

**Box A:**

```powershell
axon
> /share generate research bob --ttl-days 1
# Note the new key_id, e.g. sk_aabbccdd
> /exit
```

Use the share string on Box B to redeem. Then on Box A, **manually
edit the manifest** to fast-forward expiry (waiting 24h is impractical):

```powershell
$manifest = "$env:OneDrive\axon-smoke\AxonStore\$env:USERNAME\.shares\.share_manifest.json"
$data = Get-Content $manifest -Raw | ConvertFrom-Json
foreach ($r in $data.issued) {
    if ($r.key_id -eq "sk_aabbccdd") { $r.expires_at = "2020-01-01T00:00:00+00:00" }
}
$data | ConvertTo-Json -Depth 10 | Set-Content $manifest
```

Wait for OneDrive to sync the modified manifest to Box B (~30s).
**Box B:**

```powershell
axon
> /project switch mounts/alice_research
> /q anything
# EXPECT: PermissionError "Share 'research' expired at 2020-01-01..."
> /exit
```

**Box A — extend the share to restore access:**

```powershell
axon
> /share extend sk_aabbccdd --ttl-days 30
# EXPECT: "Share 'sk_aabbccdd' expiry updated → <future timestamp>"
> /exit
```

Wait for OneDrive sync. **Box B:** retry — the query should succeed
again.

## Step 7 — revocation propagation

**Box A:**

```powershell
axon
> /share revoke sk_aabbccdd
> /exit
```

Wait for OneDrive sync. **Box B:**

```powershell
axon
> /project switch mounts/alice_research
# EXPECT: "Mounted project 'mounts/alice_research' is not accessible:
# mount is revoked." OR an empty list (descriptor was lazily removed).
> /exit
```

## Pass criteria

A **green** smoke run satisfies all of:

- [ ] `axon --config-validate` warns about the OneDrive path on Box A.
- [ ] No `*-wal` / `*-shm` files appear in the OneDrive `bm25_index/`
      folder at any point.
- [ ] `.dynamic_graph.db` does NOT appear in the OneDrive folder; it
      DOES appear under `~/.axon/graphs/<project_id>/` on Box A.
- [ ] `.dynamic_graph.snapshot.json` and `version.json` appear in the
      OneDrive folder and sync to Box B.
- [ ] Chroma preflight raises `RuntimeError` instead of silently
      writing.
- [ ] Box B redeems and queries successfully (Step 3).
- [ ] After owner re-ingest + sync, `/project refresh` on Box B picks
      up the new content and `/q` returns updated answers (Step 4).
- [ ] Past-expiry shares are rejected on Box B's next switch + query
      (Step 6); `/share extend` restores access.
- [ ] Revocation on Box A propagates; Box B loses access on next
      switch (Step 7).

A **yellow** result (some checks pass, some don't) → file an issue,
do not block the release if only Step 5 (mid-sync race) couldn't be
naturally reproduced.

A **red** result on Steps 1–4 or 6–7 → blocks the release.

## Cleanup

```powershell
# On both boxes:
$env:AXON_STORE_BASE = ""
# Remove the test folder. OneDrive will propagate the delete to the
# other box; no need to delete on both.
Remove-Item -Recurse -Force "$env:OneDrive\axon-smoke"
# Optionally clean up the relocated graph DB on Box A.
Remove-Item -Recurse -Force "$env:USERPROFILE\.axon\graphs\<project_id>"
```

## Reporting

After running, paste the result into the release PR (or the next
"Doc Audit" memo) with:

- OS / OneDrive client version on each box
- Pass / yellow / red verdict per step
- Any `*-wal` / `*-shm` / `Conflict-` files observed
- Total elapsed time
- Anything surprising

> **Operator note:** This smoke is currently the only end-to-end
> verification that share-mount works on real OneDrive. The CI
> coverage in `tests/test_share_mount_sync_reorder.py` exercises the
> *mechanism* on synthetic local tmpdirs — useful but not equivalent.
> The server-mediated remote-mount design was explicitly rejected in
> favour of the [sealed-mount approach](../../docs/architecture/SEALED_SHARING_DESIGN.md), which
> keeps the shared-filesystem model and adds encryption-at-rest. Until
> sealed-mount lands universally, run this smoke before each release
> that touches share-mount code.
