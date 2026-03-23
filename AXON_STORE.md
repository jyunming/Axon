# AxonStore — Multi-User Sharing Guide

AxonStore is Axon's multi-user knowledge-sharing layer. It allows one user to share a project's
knowledge base with another user on the same filesystem (or shared network drive) using HMAC-signed
share keys. All shares are read-only; the sharer retains exclusive write access.

---

## 1. Overview

AxonStore requires a **shared filesystem path** that both the sharer and grantee can read. This
can be a network share, a mounted volume, or a local path on a multi-user machine.

```
┌─────────────────────────────────────────────┐
│  Shared filesystem  (/data/axon-store)       │
│                                              │
│  alice/project-a/   ← Alice's project data  │
│  bob/project-b/     ← Bob's project data    │
└─────────────────────────────────────────────┘

Alice generates a share key → sends it to Bob out-of-band (e.g. Slack, email)
Bob redeems the key → mounts alice/project-a as read-only in his Axon instance
```

---

## 2. Initialisation

AxonStore must be initialised once per user before sharing can be used.

**REST API:**
```bash
curl -X POST http://localhost:8000/store/init \
  -H "Content-Type: application/json" \
  -d '{"base_path": "/data/axon-store"}'
```

**REPL:**
```
/store init /data/axon-store
```

**VS Code:**
Run the command `Axon: Initialise Store` from the Command Palette (`Ctrl+Shift+P`).

The `base_path` is persisted in the active `config.yaml` via `config.save()` and is restored on the next `axon-api` startup.

---

## 3. Identity

Once initialised, Axon reports the current user's store identity.

**REST API:**
```bash
curl http://localhost:8000/store/whoami
# Response: {"username": "alice", "store_path": "/data/axon-store", "status": "ready"}
```

**REPL:**
```
/store whoami
```

The `username` is derived from the OS username (`os.getlogin()`).

---

## 4. Sharing a Project

The sharer generates a signed share key that encodes: sharer identity, project name, grantee
username, and a timestamp. The key is HMAC-signed with a secret derived from the store path.

**REST API:**
```bash
curl -X POST http://localhost:8000/share/generate \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "grantee": "bob"}'
# Response: {"share_string": "<base64-encoded-payload>", "key_id": "sk_a1b2c3d4", "project": "my-project", "grantee": "bob", "owner": "<your-username>"}
```

**REPL:**
```
/share generate my-project bob
```

The `share_string` (a raw base64 string) must be sent to the grantee out-of-band.
Share strings do not expire by default. All shares are **read-only** — there is no
`write_access` capability.

---

## 5. Receiving a Share

The grantee redeems the share string to mount the sharer's project locally.

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

A mount descriptor is created under `mounts/alice_my-project/mount.json` (canonical,
platform-independent record). The mount appears in `/project list` as `mounts/alice_my-project`.

To query the mounted project, switch to it using the prefixed name:
```
/project switch mounts/alice_my-project
What are the key findings?
```

---

## 6. Revoking Access

The sharer can revoke a share at any time using its `key_id`.

**REST API:**
```bash
curl -X POST http://localhost:8000/share/revoke \
  -H "Content-Type: application/json" \
  -d '{"key_id": "sk_a1b2c3d4"}'
```

**REPL:**
```
/share revoke sk_a1b2c3d4
```

Revocation marks the key as revoked in the owner's manifest (`.share_manifest.json`). The
effect on the grantee side is **active at switch time and lazy otherwise**: when the grantee
calls `POST /project/switch` targeting a mounted project, Axon validates the share against the
owner's manifest first and returns `404` immediately if the share has been revoked. Revocation
is also checked lazily on project-list and share-list operations for any mounts that were not
explicitly switched to. There is no real-time HTTP 403 on every mounted read path.

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

- **All shares are read-only.** Any attempt to ingest into a mounted project returns HTTP 403.
  There is no `write_access` parameter — it was removed in v0.9.0.
- **Shared filesystem required.** Both users must have filesystem access to the `base_path`
  set during `store init`. Network latency affects ingest performance but not query performance
  (mounted read scopes point at the mounted project's own vector and BM25 paths via the descriptor).
- **Key security.** Share strings contain a base64-encoded payload and HMAC signature. Do not
  share them over untrusted channels. Revocation is recorded in the owner's
  `.share_manifest.json`. Revocation is checked at switch time and on every retrieval; it is
  descriptor/manifest-based, not HMAC-on-access. HMAC is only used during the initial redeem step.
- **Project isolation.** Each user's projects are stored under `{base_path}/AxonStore/{username}/`.
  Users cannot read each other's data without an explicit share key.
- **No expiry by default.** Share keys do not expire. Revoke explicitly when access should end.
