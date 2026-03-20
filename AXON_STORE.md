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

The `base_path` is stored in `~/.axon/store.json` and persists across restarts.

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
# Response: {"share_string": "axon-share-v1:...", "key_id": "sk_a1b2c3d4", "expires_at": null}
```

**REPL:**
```
/share generate my-project bob
```

The `share_string` (e.g. `axon-share-v1:eyJ...`) must be sent to the grantee out-of-band.
Share strings do not expire by default. All shares are **read-only** — there is no
`write_access` capability.

---

## 5. Receiving a Share

The grantee redeems the share string to mount the sharer's project locally.

**REST API:**
```bash
curl -X POST http://localhost:8000/share/redeem \
  -H "Content-Type: application/json" \
  -d '{"share_string": "axon-share-v1:eyJ..."}'
# Response: {"mounted_as": "alice/my-project", "status": "mounted"}
```

**REPL:**
```
/share redeem axon-share-v1:eyJ...
```

The project is mounted as `<owner>/<project>` (e.g. `alice/my-project`). It appears in
`/project list` alongside local projects but is flagged as `[mount]`.

To query the mounted project:
```
/project switch alice/my-project
What are the key findings?
```

Or via CLI:
```bash
axon --project alice/my-project "What are the key findings?"
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

Revocation is **immediate**. The grantee's next query against the mounted project returns
HTTP 403. The mount entry remains in the grantee's project list but all reads fail until
the mount is removed.

---

## 7. Listing Shares

View all outgoing shares (grants you created) and incoming mounts (shares you redeemed).

**REST API:**
```bash
curl http://localhost:8000/share/list
# Response:
# {
#   "outgoing": [{"key_id": "sk_a1b2c3d4", "project": "my-project", "grantee": "bob", "revoked": false}],
#   "incoming": [{"mounted_as": "carol/research", "owner": "carol", "revoked": false}]
# }
```

**REPL:**
```
/share list
```

Revoked entries are shown with `"revoked": true` so you can identify stale mounts to remove.

---

## 8. Operational Notes

- **All shares are read-only.** Any attempt to ingest into a mounted project returns HTTP 403.
  There is no `write_access` parameter — it was removed in v0.9.0.
- **Shared filesystem required.** Both users must have filesystem access to the `base_path`
  set during `store init`. Network latency affects ingest performance but not query performance
  (query uses local vector store; only metadata is read from the share path).
- **Key security.** Share strings contain a base64-encoded payload and HMAC signature. Do not
  share them over untrusted channels. A revoked key's signature is invalid server-side; the
  store maintains a revocation list in `{base_path}/.revoked`.
- **Project isolation.** Each user's projects are stored under `{base_path}/{username}/`.
  Users cannot read each other's data without an explicit share key.
- **No expiry by default.** Share keys do not expire. Revoke explicitly when access should end.
