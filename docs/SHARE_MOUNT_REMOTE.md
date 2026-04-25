# SHARE_MOUNT_REMOTE — design for server-mediated mounts (Shape 1)

> **Status:** Design proposal (ADR). No implementation yet.
> Supersedes the file-sharing portions of [SHARE_MOUNT.md](SHARE_MOUNT.md)
> when adopted; runs alongside the existing descriptor-mount model during
> rollout.

## Context

Through `fix/share-mount-sqlite-wal-safety` (PR pending) we hardened the
existing **file-sharing** mount model — owner's project files live on a
shared filesystem (SMB / OneDrive / etc.) and grantees open them
directly. That work shipped four tightly-related fixes:

1. WAL → DELETE journal mode for the SQLite-backed components.
2. Owner-side DB relocation off cloud-sync paths.
3. `version.json` marker + `MountSyncPendingError` retry loop.
4. Share-key TTL + `extend_share_key`.

Each was correct; together they patch around a fundamental tension:
**multi-machine file sharing + active database files = pain.** Even with
all four fixes, the file-sharing model still requires:

- A shared filesystem reachable from every grantee
- Careful path classification (cloud-sync paths must be detected and
  warned about — see `src/axon/paths.py`)
- A `MountSyncPendingError` retry dance for cloud-sync mid-sync races
- Hard-cutoff TTL so revoked-but-not-noticed shares can't leak forever
- A grantee with cached binary files **can still read after revocation**
  if they have a hacked client — Option A from #54 (encrypt-at-rest with
  per-share keys) is the only thing that closes that hole, and it's a
  major engineering project

This ADR proposes a different tack — **owner runs `axon-api`; grantees
mount the URL, not the folder.** The shared filesystem disappears
entirely from the grantee's view; bytes never sit on the grantee's
disk; revocation is server-side and immediate.

The phrase that prompted this design was: *"data owner can grant the
database by key still, then users can use it as link instead of real
copy and use key to get access."* "Link instead of real copy" maps
directly onto "URL + bearer key."

## Decision

Add a second mount **type** alongside the existing descriptor-backed
file mount. The new type, **`remote`**, stores an HTTPS endpoint and a
bearer key in `mount.json` instead of a `target_project_dir`. When a
grantee activates a remote mount, Axon swaps in `RemoteVectorStore` /
`RemoteBM25Retriever` adapters that proxy queries to the owner's
`axon-api` instead of opening local files.

The existing file-mount path is kept intact so there is no breaking
change for users on a healthy SMB share or a single-machine setup.

### Mount descriptor schema (v2)

`mount.json` gains an optional `mount_type` field. Existing descriptors
are read as `mount_type: "descriptor"` (current behaviour).

```json
{
  "mount_name": "alice_research",
  "mount_type": "remote",
  "owner": "alice",
  "project": "research",
  "api_url": "https://alice-axon.tail-1234.ts.net:8000",
  "bearer_key": "sk_a1b2c3d4...full token here...",
  "share_key_id": "sk_a1b2c3d4",
  "redeemed_at": "2026-04-25T12:34:56Z",
  "state": "active",
  "revoked": false,
  "revoked_at": null,
  "readonly": true,
  "descriptor_version": 2
}
```

Differences from `descriptor` type:
- `target_project_dir` is replaced by `api_url` + `bearer_key`.
- `bearer_key` is stored privately (file mode 600 like
  `.share_keys.json`) — it is the auth credential, not just a record.

### Adapters

Two new modules, both implementing the existing minimal interfaces:

`src/axon/remote/remote_vector_store.py`:

```python
class RemoteVectorStore:
    """Thin proxy that forwards search/get_by_ids to owner's axon-api."""
    def __init__(self, api_url: str, bearer_key: str, project: str): ...
    def search(self, query_vec, top_k, filters) -> list[dict]: ...
    def get_by_ids(self, ids) -> list[dict]: ...
    def list_documents(self) -> list[dict]: ...
    # add() / delete_by_ids() raise PermissionError — mounts are read-only
```

`src/axon/remote/remote_bm25.py` mirrors the BM25Retriever interface
the same way.

Both use a small `httpx.Client` with retries, exponential backoff, and
a per-mount circuit breaker so a flapping owner doesn't lock up the
grantee's REPL.

### Mount resolution change

`AxonBrain.switch_project()` (currently at `src/axon/main.py:1150–1175`)
gets a third branch:

```python
if mount_type == "remote":
    self.vector_store = RemoteVectorStore(desc["api_url"],
                                          desc["bearer_key"],
                                          desc["project"])
    self.bm25 = RemoteBM25Retriever(desc["api_url"],
                                    desc["bearer_key"],
                                    desc["project"])
    # Skip _close()/GC of local handles — there are none.
elif mount_type == "descriptor":
    # existing file-mount path (unchanged)
```

### Share-key flow change

`generate_share_key` gains an `endpoint_mode` parameter:

```python
result = generate_share_key(
    owner_user_dir, project, grantee,
    ttl_days=30,
    endpoint_mode="remote",            # "descriptor" (default) | "remote"
    api_url="https://alice-axon.tail-1234.ts.net:8000",
)
```

The returned `share_string` then encodes:
`base64(key_id:token:owner:project:owner_store_path:endpoint_mode:api_url)`

`redeem_share_key` reads `endpoint_mode` and writes either a
`descriptor`-type or `remote`-type `mount.json`.

Backward compat: legacy share strings (5 fields, no endpoint_mode)
default to `descriptor` mode.

### Authentication model

- Bearer token = the share key's `token` field, presented as
  `Authorization: Bearer <token>` on every request.
- Owner's `axon-api` validates the token against
  `.share_keys.json`'s `issued` records — same HMAC + revocation +
  expiry checks used in `redeem_share_key` today.
- 401 = unknown / revoked / expired key. Grantee surfaces this via
  the existing "share has been revoked" UX in `_check_mount_revocation`.
- 5xx + retry-after = transient owner outage. Adapters retry with
  exponential backoff; circuit-breaker opens after N consecutive
  failures.
- TLS is **mandatory** in v1. `api_url` must be `https://`. We will
  not accept `http://` to avoid putting bearer tokens on the wire in
  plaintext. (Local-LAN dev exception: allow `http://` only when the
  hostname resolves to RFC1918 + a `--insecure-local-dev` flag is set.)

### Network model — how does the grantee reach the owner's API?

This is the operational cost of Shape 1. Three supported deployment
patterns, each documented in SETUP.md:

| Pattern | Setup cost | When to use |
|---|---|---|
| **Internal LAN** | Owner runs `axon-api --host 0.0.0.0`; firewall rule on owner's box | Office network, single subnet |
| **Tailscale / WireGuard** (recommended) | Both sides install Tailscale; owner gets a stable `*.tail-NNNN.ts.net` URL | Remote teams, no port-forwarding |
| **Public reverse proxy** (Cloudflare / ngrok / Caddy) | Owner exposes axon-api behind a TLS proxy with a long-lived URL | Internet-facing, anyone-can-reach |

Owner's API process must stay running for grantees to query — that's
the genuine new dependency Shape 1 introduces vs the file-mount model
where the owner could be offline.

## Consequences

### What gets simpler

- Grantees never touch a shared filesystem → SQLite-WAL safety,
  cloud-sync path classification, and `version.json` refresh become
  irrelevant for the grantee path. (Owner-side hardening still
  matters — the owner has local files.)
- Revocation is **server-side and immediate** — no caching window, no
  TTL needed for the security ceiling. (TTL is still useful as an
  optional auto-rotation policy.)
- Cached-bytes-after-revocation attack disappears entirely — grantees
  hold no data, only a URL + key.
- Cross-platform parity is automatic — works on macOS, Linux, Windows
  identically because the wire protocol is HTTPS.

### What gets harder

- Owner must run a long-lived `axon-api` process reachable from the
  grantee.
- Grantee performance is bounded by network latency to owner — every
  query is a round-trip. (Local file mounts are local-disk fast.)
- Offline grantees cannot query at all. (File-mount grantees with
  cached/synced files can.)
- New attack surface: the `axon-api` endpoint is now an authenticated
  internet service. Needs basic hardening (rate limit, TLS, auth log).

### What is unchanged

- Owner-side ingest, governance, dynamic-graph backend — all the same.
- Existing file-mount descriptors keep working — no migration forced.
- `version.json` continues to be useful for owner-side debugging.
- Share-key generation/revocation API stays the same shape; only the
  share-string payload grows by 2 fields.

## Cross-interface plan

| Surface | Change |
|---|---|
| **REST** | New `endpoint_mode` + `api_url` fields on `POST /share/generate`. New `RemoteVectorStore` proxy hits the owner's existing `POST /query` etc. — no new endpoints needed on the owner side. |
| **MCP** | `share_project` gains `endpoint_mode` + `api_url` kwargs. No new tool. |
| **REPL** | `/share generate <project> <grantee> [--ttl-days N] [--remote <url>]`. Without `--remote`, falls back to descriptor mode (today's behaviour). |
| **CLI** | `--share-generate PROJECT GRANTEE` gains `--remote URL` flag. |
| **VS Code** | Inherits via REST; share-creation panel gets a "Remote URL (optional)" field. |

## Migration / coexistence

- Both mount types live in the same `mounts/` directory. `mount_type`
  field disambiguates.
- A grantee can have a mix — some `descriptor` (file-shared), some
  `remote`.
- Share-strings generated before this lands continue to work
  (parsed as 5-field legacy format → descriptor mode).
- No forced migration; users can re-issue shares as remote whenever
  they choose.

## Trade-offs vs alternatives

| Approach | Network dep | Offline grantee | Bytes on grantee | Revocation | Cloud-sync corruption risk | Implementation cost |
|---|---|---|---|---|---|---|
| **File mount today** (post Path A) | None (just FS) | ✅ Works | ✅ Has copy | Lazy / TTL | Mitigated, not eliminated | Already done |
| **Shape 1: remote mount** (this ADR) | Always | ❌ No access | ❌ Never has copy | Server-side, immediate | Eliminated | Medium |
| **Shape 2: encrypted-at-rest** (#54 Option A) | None | ✅ Works | ✅ Encrypted copy | Key rotation invalidates cached | Mitigated, encryption layer is the safety | Large |
| **Path B: event-log replay** (research synthesis) | Pull on poll | ✅ Works | ✅ Local rebuild | Lazy / TTL | Eliminated | Medium-large |

Shape 1 is the smallest change that actually closes the
cached-bytes-after-revocation gap. Shape 2 is more expressive (works
offline) but materially harder.

## Open questions

1. **Latency budget.** What's the round-trip ceiling before grantee
   experience degrades? Need a benchmark before we decide whether to
   cache query results or shipped chunk metadata grantee-side.
2. **Batch queries.** When a grantee uses HyDE / multi-query / RAPTOR
   features that issue many parallel calls, does the owner-side API
   become a bottleneck? Consider a `/query/batch` endpoint.
3. **Streaming responses.** `query_stream` over HTTP — long-lived SSE
   or chunked transfer? Both work; SSE is simpler.
4. **Owner-side auth log.** Every grantee request should be visible in
   the governance audit trail. Map share key → grantee identity in the
   `actor` field.
5. **Bearer key rotation.** When a key expires (`extend_share_key`
   wasn't called in time), what does the grantee see? Probably 401 with
   a hint message; UX is the same as a revoked key.
6. **Per-share rate limit.** A misbehaving grantee shouldn't be able
   to DoS the owner. Token-bucket per `share_key_id`?
7. **Caching grantee-side.** Should `RemoteVectorStore` cache hot
   chunks locally? If yes, the cached-bytes-after-revocation hole
   reappears for the cache window. Probably opt-in only, off by default.
8. **TLS certs.** Tailscale gives you a cert automatically. For
   bring-your-own-domain users, do we ship a Let's Encrypt
   integration or expect a reverse proxy in front?

## Out of scope for v1

- Encryption-at-rest (Shape 2). Tracked as a separate future epic.
- Peer-to-peer (no owner-side server). Would require WebRTC or
  similar — much bigger scope.
- Multi-owner / consensus (Shared Drives style). One owner per
  project remains the canonical model.
- Write access (RW mounts). Mounts are read-only by design; this ADR
  preserves that.

## Implementation phases

1. **Phase 1 (foundations, no behaviour change yet):** add
   `mount_type` field to descriptor schema, write `RemoteVectorStore`
   + `RemoteBM25Retriever` skeletons, update share-string parser to
   tolerate the new fields. Tests for the schema + parser only.
2. **Phase 2 (end-to-end happy path):** wire `switch_project` to
   instantiate `RemoteVectorStore` when `mount_type == "remote"`. Add
   `endpoint_mode` + `api_url` to `share_generate` REST/MCP/REPL.
   Manual two-machine test with Tailscale.
3. **Phase 3 (failure modes):** retry / circuit breaker, 401 → revoked
   UX, TLS enforcement, optional `--insecure-local-dev`.
4. **Phase 4 (UX):** REPL completion, VS Code panel field, docs in
   SETUP.md + SHARE_MOUNT.md (cross-link this ADR).
5. **Phase 5 (audit):** governance log entries with grantee identity,
   `axon doctor` reports active remote mounts, per-share rate limit.

Each phase is one PR. Phase 1 is the only piece that has zero
behaviour change and could land speculatively; Phases 2–5 each
deliver visible value.

## Pre-PR checklist (when this ADR is greenlit)

- [ ] Open GitHub issue "share-mount remote: Phase 1 — schema +
      adapter skeletons"
- [ ] Write a `RemoteVectorStore` performance benchmark (vs local
      `OpenVectorStore`) so we have a baseline
- [ ] Decide on the TLS-mandatory rule (or accept the
      `--insecure-local-dev` escape hatch)
- [ ] Decide whether grantee-side caching is opt-in (recommended) or
      forbidden (safer)
- [ ] Pick a default network model to recommend in SETUP.md (lean:
      Tailscale)
