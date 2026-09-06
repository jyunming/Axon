# Axon's response to "What FrontierInsight will need from Axon"

**Status:** assessment, not a commitment. Answers the four questions in
[FRONTIERINSIGHT_SKILL_MEMORY.md](FRONTIERINSIGHT_SKILL_MEMORY.md) §7 with
evidence from the current tree, and proposes a different order of work than the
proposal implies.

Written 2026-09-05, against Axon `4934980`.

---

## Summary

The four requirements are not equally expensive, and the cheapest one is the one
the proposal leads with as most consequential.

| § | Requirement | Axon-side cost | Verdict |
|---|---|---|---|
| 5 | Skill knowledge exempt from staleness | **none** | The concern does not apply — nothing deletes |
| 2 | `skill` naming collision | **none (docs only)** | Axon should give the word up, not FI |
| 4 | Provenance survives retrieval | **small** | Real gap, and a **prerequisite for §3** |
| 3 | Many-project retrieval in one call | **medium** | Half of it already exists; the missing half is the awkward half |

Recommended order: **§4 → §2 → §3**, with §5 needing nothing. The proposal's
implied order (§3 first, §4 as a "probably fine" check) would deliver merged
results FI cannot attribute, which defeats the distillation use case.

---

## §5 — Staleness: no change needed, and the worry is misplaced

`get_stale_docs` is a **read-only report**. There is no retention job, no sweep,
and nothing anywhere in Axon that deletes documents on an age basis. Deletion is
only ever explicit (`delete_documents`, `clear_knowledge`, `delete_project`).

So the stated fear — "no retention job silently removes the substrate the skill
library is built on" — has nothing to act on. **FI needs no exemption mechanism
because there is nothing to be exempt from.**

There is, however, a different problem worth knowing about, in the opposite
direction. The report is built from `_source_hashes`
(`src/axon/api_routes/ingest.py:351`), an **in-process, in-memory dict that
resets on every server restart** — the MCP tool's own docstring says so
(`src/axon/mcp_server.py:295`). A skill corpus ingested last week will simply not
appear in the report, no matter how stale it is.

If FI ever wants staleness *signals* over a long-lived corpus, the current
mechanism cannot provide them for any project, skill-related or not. That is a
separate piece of work and not one this proposal needs to trigger.

---

## §2 — Naming: Axon should give the word up

The proposal offers for FI to rename, on the grounds that Axon's usage is older
and published. Reasonable, but the assessment from this side is that Axon's usage
is the weaker claim:

- `docs/SKILLS.md` (319 lines) and `docs/MCP_TOOLS.md` (635 lines) document the
  **same MCP tools**, both with per-tool sections (`### query_knowledge`,
  `### search_knowledge`, …). This is duplication, not two documents.
- `docs/SKILLS.md:12` states "all 30 Axon tools". The real count is 56, and is
  about to become roughly 13 — the agent tool surface is mid-restructure
  precisely because 57 tools cost ~10.5k tokens of agent context per turn.
- "Skill" was never an accurate name for what those entries are. They are tools.

So: Axon plans to retire `docs/SKILLS.md` during its documentation consolidation
and stop using "skill" for MCP tools. That removes a duplicate document, fixes a
stale count, and frees the word — one change resolving three problems.

**FI should keep "skill".** No rename needed on that side.

---

## §4 — Provenance: a real gap, and it must come first

Two concrete defects, both in the cross-project path FI would rely on:

**1. Chunk metadata never records the originating project.** Nothing in the
ingest path writes a project into chunk metadata. A retrieved chunk therefore
cannot say which project it came from, because that fact was never stored.

**2. Cross-project fan-out silently drops colliding documents.**
`MultiVectorStore.search` (`src/axon/vector_store.py:888-890`) merges sub-store
results keyed on `doc["id"]`, keeping the higher score:

```python
doc_id = doc["id"]
if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
    seen[doc_id] = doc
```

Two projects holding a document with the same id — which is likely, given ids
derive from source paths, and quests reusing the same software will reuse
filenames — collapse to one result. The other disappears with no signal. It also
does not tag results with the sub-store they came from.

For FI's stated purpose this is the load-bearing gap: "which projects taught me
this?" and "one lucky quest versus six" are both unanswerable, and the second
defect actively destroys the evidence that would answer them.

**This is why §4 must precede §3.** Shipping many-project retrieval on top of a
merge that neither attributes nor preserves per-project results would hand FI a
result set that looks right and cannot be reasoned about. The fix is small —
carry the origin project through the fan-out and key dedup on (project, id)
rather than id alone — but it is not optional if §3 is going to be useful.

---

## §3 — Many-project retrieval: half-built, and the missing half is the awkward one

More of this exists than the proposal assumes, and the part that is missing is
not the part that looks hard.

**What exists.** `MultiVectorStore` and `MultiBM25Retriever`
(`src/axon/vector_store.py:856`, `:961`) already fan out across several stores
concurrently, merge, and re-rank server-side. This is not a sketch — it is the
path that runs today whenever a parent project is active, querying across all its
descendants. `_switch_to_scope` (`src/axon/main.py:810-829`) already builds those
wrappers from an **arbitrary list of project paths**, not from a hardcoded
hierarchy.

So the ranking quality the proposal worries about losing to client-side fan-out
(option 1) is already available server-side.

**What is missing.** That path is a *mode switch*, not a query. It calls
`self.close()` and rebuilds the brain's stores
(`src/axon/main.py:808-829`), mutating global state. Driving it per-query would
hit exactly the hazard `switch_project`'s own MCP docstring warns about:

> **WARNING: This mutates global server state. Do not call from concurrent
> request handlers.**

The work is therefore: construct a read-only fan-out over a caller-supplied set
of projects **without** tearing down and rebuilding the active brain's stores,
then widen the `project` parameter to accept a list. Medium, not small — but well
short of building fan-out retrieval from nothing.

**On option 3 (a first-class project group):** FI's hunch is right that it
overlaps with existing machinery. A parent project already *is* a named,
persistent set of projects that behaves as one retrieval namespace. If FI's
quests can be arranged as children of a parent, option 3 partially exists today —
worth checking against FI's project layout before treating it as new work.

---

## Answers to §7

1. **Is many-project retrieval something Axon wants independent of FI?**
   Partly — it already exists for project hierarchies, so the concept is not
   FI-specific. Making it work for arbitrary sets, per-call and without a state
   switch, is not currently on Axon's roadmap and would be scheduled on FI's
   need. Not a no; not yet a yes.

2. **Do retrieved chunks already carry origin project + ingest metadata?**
   **No.** The project is never written into chunk metadata, and the
   cross-project merge neither tags origin nor preserves same-id documents from
   different projects. See §4.

3. **Is there an existing convention for documents maintenance should leave
   alone?** There is no maintenance that touches documents, so no convention is
   needed. See §5.

4. **Does the naming collision matter?** It does, and Axon will resolve it by
   dropping its own use of "skill". FI keeps the word. See §2.

---

## What this does not commit to

No timeline. Axon is mid-way through a slim-down (its install dropped 2.3 GB, its
dead code and one dead surface are gone, and its agent tool surface is about to
shrink by roughly 75%), and taking on cross-project retrieval before that lands
would re-grow the surface the slim-down exists to reduce.

The §4 provenance fix is small enough to be worth doing on its own merits —
silently dropping colliding documents is a defect regardless of FI — and is the
natural first step whenever this is picked up.
