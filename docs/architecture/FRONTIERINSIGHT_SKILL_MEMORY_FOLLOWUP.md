# FI follow-up: three corrections, verified against the tree

**Status:** findings, not a request. Responds to
[FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md](FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md).

The response was checked rather than accepted — its §4 and §5 conclusions both
change what FI builds, so they were worth verifying independently. Two of the
three answers hold exactly as written. One does not, and it is the one that set
the recommended order of work.

Written 2026-09-05, against Axon `4934980`.

---

## Summary

| Claim in the response | Verdict |
|---|---|
| §5 — nothing deletes on age, no exemption needed | **Confirmed** |
| §2 — Axon gives up the word "skill" | **Accepted with thanks**; FI keeps it |
| §4 defect 1 — project not in chunk metadata | **True, but origin is recoverable anyway** |
| §4 defect 2 — cross-project same-id collapse | **Already fixed in this tree** |
| "§4 must precede §3" | **No longer holds** — the defect it rested on is gone |

Plus one unrelated defect found while testing: `POST /add_text` returns 500 on
the running server.

---

## §4 defect 2 is already fixed, with a comment naming this exact concern

The response cites `MultiVectorStore.search` merging on `doc["id"]`
(`src/axon/vector_store.py:887-890`) and concludes that two projects holding a
same-id document collapse to one result, destroying the evidence FI needs.

The merge code is exactly as quoted. But documents cannot reach it with colliding
ids, because ids are namespaced at ingest — `src/axon/main.py:2429-2441`:

```python
# Namespace chunk IDs by project to prevent cross-project collisions in
# MultiVectorStore / MultiBM25Retriever dedup. The default project is left
# unchanged so existing single-project deployments are unaffected.
if self._active_project and self._active_project != "default":
    _ns = get_project_id(self._active_project) or self._active_project
    _ns_prefix = f"{_ns}::"
    for _doc in documents:
        if not _doc["id"].startswith(_ns_prefix):
            _doc["id"] = _ns_prefix + _doc["id"]
```

The comment names the same two classes the response identifies as the failure
site. This was fixed deliberately, and the fix predates the analysis.

Observed live rather than only read — a `search_knowledge` against
`autoresearch_eval` returns ids of the form:

```
proj_51d42033ec5d413694db19f25be91270::file_98c8ff97423c6945e77b6ea3_p5_chunk_2
```

Two projects therefore produce disjoint id spaces, and the `seen[doc_id]` merge
has nothing to collapse.

### The residual, which is real but narrow

That same block **exempts `default`** by design. So a collision remains possible
between `default` and any other project, since `default`'s ids carry no prefix
and another project's chunk could in principle share the bare id. Worth knowing;
much smaller than the general defect described, and it does not affect FI, which
does not use `default`.

---

## §4 defect 1 is true as stated, but attribution already works

There is indeed no `project` key written into chunk metadata. Verified: the only
occurrence of `"project":` in the ingest module is inside the `get_stale_docs`
report builder (`src/axon/api_routes/ingest.py:361`), not the write path.

But the origin project is recoverable regardless, because the id prefix *is* the
project id — `get_project_id(project)` produces the `proj_…` value that appears
in every namespaced id. So "which project taught me this?" is answerable today
by mapping the prefix back through `get_project_id`.

That makes a `project` metadata field **ergonomics rather than capability**. FI
would still prefer it — parsing an id prefix to recover provenance is the kind of
thing that breaks quietly when an id format changes — but it is a convenience
ask, not a blocker, and FI is content to do the mapping itself in the meantime.

### Metadata does survive retrieval, richly

The open question from the original proposal is answered. A retrieved chunk
carries `source` (full path), `source_id` (also namespaced), `chunk`,
`chunk_index`, `total_chunks`, `chunk_kind`, `dataset_type`, `subdoc_locator`,
`type`, and `parent_text`. Nothing FI writes at ingest appears to be dropped.

---

## §5 confirmed, and the deeper point stands

No age-based deletion exists; `get_stale_docs` is a report. The requirement is
withdrawn from FI's roadmap.

The response's own observation — that the report reads `_source_hashes`, an
in-process dict that resets on restart, so a corpus ingested last week never
appears regardless of age — is the more useful half of that section, and FI has
recorded it as a known limitation rather than a request.

---

## The ordering conclusion no longer follows

The response recommends §4 → §2 → §3, on the grounds that shipping many-project
retrieval over a merge that "neither attributes nor preserves per-project
results" would hand FI results it cannot reason about.

Since the merge *does* preserve per-project results (disjoint id spaces) and
origin *is* recoverable (id prefix), that argument no longer holds. §3 is not
blocked by §4.

**FI is not asking for §3 to be scheduled.** The opposite, in fact — see below.

---

## FI no longer needs many-project retrieval at all

A correction to FI's own proposal, which asserted that "each quest is naturally
its own project, so the query is inherently many-project."

FI does not work that way. `core/knowledge.py:189` pins a single project —
`frontier-insight`, overridable via `FI_AXON_PROJECT` — and `:2518` switches to
it. Every quest writes into that one project. A single-project query already
spans every quest.

So the expensive requirement in the original proposal was premised on a layout FI
does not have. What FI actually needs is per-quest attribution *within* one
project, which is document metadata rather than a project boundary — and FI
already writes it (`add_quest_artifacts(quest_id, …)`, plus a `fi_topic_event`
keyed by topic slug).

Apologies for the wasted analysis on §3. The response's instinct — "worth
checking against FI's project layout before treating it as new work" — was
right, and pointed at the flaw from the outside before FI found it from within.

If FI ever moves to project-per-quest, the note that a parent project already
behaves as one retrieval namespace over its children is the useful part to
retain, and `subs/` with `parent/child` naming (`src/axon/projects.py:213-223`)
is where that would start.

---

## Unrelated: `POST /add_text` returns 500 on the running server

Found while attempting a metadata probe, before switching to the read-only
method above.

```
POST http://127.0.0.1:8420/add_text
{"detail":"[Errno 22] Invalid argument"}
```

Reproduced with and without a `metadata` object, and against two different
active projects (`kgtest`, `default`), so it is not payload- or
project-specific. The MCP `ingest_text` tool surfaces it as a 500.

**Caveat:** the server process under test was started before the current working
tree, which has uncommitted changes on `chore/pr3-drop-splade-test-federated`.
This may already be fixed in the tree, or may be specific to this install. No
traceback was recoverable — the API log records request lines only, and
`api-8420.err.log` is from an earlier run.

Flagged rather than diagnosed; it is outside what this thread is about.

---

## Net effect on FI's roadmap

- §5 requirement withdrawn.
- §2 accepted — FI keeps "skill", with thanks.
- §4 downgraded from blocker to convenience ask.
- §3 withdrawn — FI's single-project layout does not need it.

Which leaves nothing that Axon must schedule on FI's behalf. The `default`-prefix
residual and the `/add_text` failure are Axon's own to weigh; neither is an FI
request.
