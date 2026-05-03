# AxonStore — Multi-User Sharing Guide

AxonStore is Axon's multi-user knowledge-sharing layer. It allows one user to share a project's knowledge base with another user on the same filesystem (or shared network drive) using HMAC-signed share keys. All shares are read-only; the sharer retains exclusive write access.

---

## 1. Overview

AxonStore is always active. By default your data lives at `~/.axon/AxonStore/<username>/`. On first run Axon creates this layout automatically — no manual initialisation step required for single-user use. To move the store to a shared drive (for multi-user sharing), see [Section 2 — Changing the Store Base Path](#2-changing-the-store-base-path).

**Check store status at any time:**

```bash
# REST
curl http://localhost:8000/store/status
# MCP tool
get_store_status()
# Returns: {"initialized": true, "path": "~/.axon/...", "store_version": 2, ...}
# or:      {"initialized": false, ...} on a completely fresh install
```

For cross-user sharing, point `store.base` at a **shared filesystem path** that all participants can read — a network share, mounted volume, or local path on a multi-user machine.

```
┌──────────────────────────────────────────────────────┐
│  Shared filesystem  (/data/axon-store)                │
│                                                       │
│  AxonStore/                                           │
│    alice/project-a/   ← Alice's project data         │
│    bob/project-b/     ← Bob's project data           │
└──────────────────────────────────────────────────────┘
Alice generates a share key → sends it to Bob out-of-band (e.g. Slack, email)
Bob redeems the key → mounts alice/project-a as read-only in his Axon instance
```

---

## 2. Changing the Store Base Path

By default Axon uses `~/.axon` as the store base. To move your data to a shared drive, call `/store/init` with the new path. This changes the base for the current session and optionally persists it to `config.yaml`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | string | required | Absolute path to the new store base directory |
| `persist` | bool | `false` | Write the new path to `config.yaml` permanently |

**REST API:**

```bash
curl -X POST http://localhost:8000/store/init \
  -H "Content-Type: application/json" \
  -d '{"base_path": "/data/axon-store", "persist": true}'
```

**REPL** (persists automatically):

```
/store init /data/axon-store
```

**VS Code:**

Run `Axon: Initialise Store` from the Command Palette (`Ctrl+Shift+P`). Persists automatically.

**Permanent via config.yaml** (recommended for shared deployments):

```yaml
store:
  base: /data/axon-store   # or set AXON_STORE_BASE env var
```

> **Note:** Changing the base path makes any projects at the previous path temporarily unreachable. The response includes a `warning` and `unreachable_projects` list if this applies:
> ```json
> {
>   "status": "ok",
>   "warning": "2 project(s) will be unreachable until the previous store path is restored: old-project, legacy-proj",
>   "unreachable_projects": ["old-project", "legacy-proj"]
> }
> ```

---

## 3. Identity

Check your current store identity and which path is active.

**REST API:**

```bash
curl http://localhost:8000/store/whoami
# Response: {"username": "alice", "store_path": "/data/axon-store/AxonStore", "user_dir": "/data/axon-store/AxonStore/alice"}
```

**REPL:**

```
/store whoami
```

The `username` is your OS username. `store_path` is the `AxonStore/` directory inside your base. `user_dir` is your personal namespace within it.

---

## 4. Sharing a Project

Two share modes ship in v0.3.x:

- **Plaintext shares** (`sk_xxxxxxxx` key IDs) — HMAC-signed, used for unsealed projects on a shared filesystem
- **Sealed shares** (`ssk_xxxxxxxx` key IDs, `SEALED1:`-prefixed envelope) — AES-256-GCM at rest with a per-grantee AES-KW wrap; safe through OneDrive / Dropbox / Google Drive (cloud sees ciphertext only)

`/share/generate` automatically picks the right mode based on whether the project is sealed (`axon --project-seal <project>`).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | required | Name of the project to share |
| `grantee` | string | required | OS username of the recipient |
| `ttl_days` | int \| null | `null` | Positive integer days from creation until the share expires. `null` = no expiry. **v0.4.0:** honoured by both sealed (`ssk_`) and plaintext (`sk_`) shares. For sealed shares this writes an Ed25519-signed expiry sidecar next to the wrap; the grantee's mount fails with `ShareExpiredError` after expiry and auto-destroys the local DEK + cache + mount descriptor. `ttl_days <= 0` returns 422. |

> See [SHARING.md](SHARING.md) for the full sealed-share threat model, sync-folder prerequisites, and the `pip install "axon-rag[sealed]"` (or `[starter]`) extras.

**REST API:**

```bash
# Plaintext share (project not sealed)
curl -X POST http://localhost:8000/share/generate \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "grantee": "bob"}'
# Response: {"share_string": "eyJ...", "key_id": "sk_a1b2c3d4", "project": "my-project", "grantee": "bob", "owner": "<your-username>", "expires_at": null}

# Sealed share (project sealed) — same call, sealed envelope returned.
# v0.4.0: ttl_days is honoured for sealed shares via signed expiry sidecar.
curl -X POST http://localhost:8000/share/generate \
  -H "Content-Type: application/json" \
  -d '{"project": "research", "grantee": "alice", "ttl_days": 30}'
# Response: {"share_string": "U0VBTEVE...", "key_id": "ssk_a4f9c1d2", "project": "research", "grantee": "alice", "owner": "<your-username>", "security_mode": "sealed_v1", "expires_at": "2026-06-03T01:00:00Z"}
```

**REPL:**

```
/share generate my-project bob
/share generate research alice --ttl-days 30   # with expiry
```

The `share_string` is a base64 envelope:
- Plaintext: base64 of an HMAC-signed payload
- Sealed (no TTL, store locked at mint, or any pre-v0.4.0 share): base64 of `SEALED1:<key_id>:<token_hex>:<owner>:<project>:<store_path>`
- Sealed (v0.4.0, store unlocked at mint with `ttl_days`): base64 of `SEALED2:<key_id>:<token_hex>:<owner>:<project>:<store_path>:<owner_pubkey_hex>` — carries the Ed25519 verifier so the grantee can validate the expiry sidecar without owner round-trip

Send it out-of-band (Signal, encrypted email — never the same channel as the data). All shares are **read-only** — there is no `write_access` capability.

### Extending an existing share

Call `POST /share/extend` (or REPL `/share extend <key_id> --ttl-days N`) to push out the expiry of a share that's already issued — useful when a contractor's engagement is renewed without re-issuing keys. **Plaintext shares (`sk_`) only.** For sealed (`ssk_`) shares, the expiry lives in an Ed25519-signed sidecar; extending it would require re-signing under the owner's signing key and re-uploading. The supported flow for renewing a sealed share is to mint a fresh one with a longer `ttl_days` and revoke the old `key_id` once the grantee has switched over. Calling `/share/extend` on an `ssk_` key is a no-op or 422 (depending on whether the legacy manifest contains the key) — the call never modifies the signed sidecar.

---

## 5. Receiving a Share

The grantee redeems the share string to mount the sharer's project locally.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `share_string` | string | required | The full base64 share string received from the sharer |

**REST API:**

```bash
curl -X POST http://localhost:8000/share/redeem \
  -H "Content-Type: application/json" \
  -d '{"share_string": "eyJ..."}'
# Response: {"mount_name": "alice_my-project", "owner": "alice", "project": "my-project", "descriptor": {...}}
```

**REPL:**

```
/share redeem eyJ...
```

A mount descriptor is created under `mounts/alice_my-project/mount.json` (canonical, platform-independent record). The mount appears in `/project list` as `mounts/alice_my-project`.

To query the mounted project, switch to it using the prefixed name:

```
/project switch mounts/alice_my-project
What are the key findings?
```

---

## 6. Revoking Access

Two revoke flavors:

- **Soft revoke** (default) — deletes the share's wrap files. Fresh redeems fail; cached DEKs on the grantee side keep working until rotate.
- **Hard revoke** (`--rotate` / `rotate=true`) — rotates the project DEK, re-encrypts every content file, selectively re-wraps the new DEK for surviving grantees with a `.kek` sidecar. The revoked grantee's cached DEK no longer matches the ciphertext; their next query fails.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_id` | string | required | Key ID from `/share/list`. Format: `sk_xxxxxxxx` (plaintext) or `ssk_xxxxxxxx` (sealed) |
| `project` | string | required for sealed | Project name. **Required** for sealed shares (`ssk_` prefix); optional for plaintext |
| `rotate` | bool | `false` | Hard revoke — rotate DEK + re-encrypt + selective re-wrap |

**REST API:**

```bash
# Soft revoke a plaintext share
curl -X POST http://localhost:8000/share/revoke \
  -H "Content-Type: application/json" \
  -d '{"key_id": "sk_a1b2c3d4"}'

# Hard revoke a sealed share — rotate DEK + invalidate cache
curl -X POST http://localhost:8000/share/revoke \
  -H "Content-Type: application/json" \
  -d '{"key_id": "ssk_a4f9c1d2", "project": "research", "rotate": true}'
```

**REPL:**

```
/share revoke sk_a1b2c3d4
/share revoke ssk_a4f9c1d2 --project research --rotate
```

Revocation effects:
- **Plaintext share**: marks the key as revoked in the owner's manifest (`.share_manifest.json`). Validated lazily — at `POST /project/switch` and on `/project list` / `/share list` operations. There is no per-read HTTP 403.
- **Sealed share, soft**: deletes `.security/shares/<key_id>.wrapped` AND `<key_id>.kek` so fresh redeems fail. Cached DEKs in grantees' OS keyrings keep working.
- **Sealed share, hard (`rotate=true`)**: re-encrypts the full project, returns an `invalidated_share_key_ids` list so the owner knows which surviving grantees need fresh shares (legacy projects without `.kek` sidecars).

---

## 7. Listing Shares

View all outgoing shares (grants you created) and incoming mounts (shares you redeemed).

**REST API:**

```bash
curl http://localhost:8000/share/list
# Response:
# {
#   "sharing": [{"key_id": "sk_a1b2c3d4", "project": "my-project", "grantee": "bob", "revoked": false, "created_at": "..."}],
#   "shared":  [{"key_id": "sk_...", "owner": "carol", "project": "research", "mount": "carol_research", "redeemed_at": "..."}]
# }
```

**REPL:**

```
/share list
```

Revoked entries in `sharing` are shown with `"revoked": true`. Entries in `shared` do not include a `revoked` field — revoked mounts are removed lazily from the grantee's list on the next project list or switch operation.

---

## 8. Operational Notes

- **All shares are read-only.** Any attempt to ingest into a mounted project returns HTTP 403. There is no `write_access` parameter.

- **Shared filesystem required for plaintext.** Sealed shares can ride OneDrive / Dropbox / Google Drive (cloud sees ciphertext only). Plaintext shares require both users to have filesystem access to the same `base_path` set during `store init`.

- **Key security.** Sealed share strings begin (after base64-decode) with `SEALED1:<key_id>:<token_hex>:<owner>:<project>:<store_path>`; plaintext share strings are HMAC-signed payloads. Do not share them over untrusted channels (Signal / encrypted email recommended). Revocation is recorded in the owner's `.share_manifest.json` and validated lazily — at switch time, project-list, and share-list operations.

- **Project isolation and hierarchy.** Each user's projects live under `{base_path}/AxonStore/{username}/`. Users cannot read each other's data without an explicit share key. Project names are slash-separated and **nest up to 5 levels deep** (`research/papers/2024/q3/draft`); deeper paths are rejected at validation time (`_MAX_DEPTH` in `projects.py`).

- **Mount layout.** When grantee bob redeems a share for `alice/research/papers`, it appears as `mounts/alice_research_papers` in bob's namespace (the `/` is replaced with `_` in the mount name).

- **Expiry.** Share keys do not expire by default. Pass `ttl_days` at generate time (or extend later via `POST /share/extend`) to set a finite TTL, or revoke explicitly when access should end.
