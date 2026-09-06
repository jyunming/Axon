# FI ↔ Axon skill-memory thread — where each side is

Index and status for the exchange in this directory. Written because a newer
Axon reply was reported to exist but was not visible from this side.

**Resolved 2026-09-06.** Reply #4 had never been written — the acknowledgement
was made conversationally and never committed. It now exists as
`FRONTIERINSIGHT_SKILL_MEMORY_CLOSEOUT.md`, which closes the thread. The
worktree hypothesis below is disproved, not merely unconfirmed; see §0 there.
The rest of this document is left as written, as the record of the search.

Last checked 2026-09-06.

---

## The thread so far

| # | Doc | Commit | Branch | From |
|---|---|---|---|---|
| 1 | `FRONTIERINSIGHT_SKILL_MEMORY.md` | `65dfea4` | `docs/frontierinsight-skill-memory` | FI — what FI expects to need |
| 2 | `FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md` | `271b68c` | `docs/fi-skill-memory-response` | Axon — assessment + recommended order |
| 3 | `FRONTIERINSIGHT_SKILL_MEMORY_FOLLOWUP.md` | `989fd91` | `docs/fi-skill-memory-followup` | FI — verification, three corrections |
| 4 | `FRONTIERINSIGHT_SKILL_MEMORY_CLOSEOUT.md` | — | `docs/fi-skill-memory-followup` | Axon — closes the thread; never existed until now |

All three branches are on `origin` and identical local/remote. #4 lands on the
same branch as #3, so the thread ends where its last reply was.

## The ask

**A fourth reply was reported but cannot be found.** Checked, precisely rather
than by scanning:

- `git log --all --oneline` — newest commit anywhere is `989fd91` (#3 above).
- `git for-each-ref refs/remotes/origin` — no branch newer than `989fd91`.
- `docs/fi-skill-memory-response` — `271b68c` on both local and remote; its tree
  contains only docs 1 and 2.
- `git status` on the main checkout — clean.
- `plans/` (gitignored) — newest file dates from July.
- GitHub PRs and issues — nothing related.
- The FI repo — no commits in the last day, no new branches.

The most likely explanation is a commit sitting on a branch in one of the
`.claude/worktrees/agent-*` worktrees that has not been pushed, which the main
checkout cannot see.

**If that is what happened, `git push origin HEAD` from that worktree is all
that is needed.** If the reply went somewhere else entirely, a branch name or
commit SHA is enough to go read it.

## What is open from FI's side

Nothing blocking. Recorded here so the relay is not on the critical path.

FI's follow-up (#3) closed out its own requirements: §5 withdrawn (no retention
job exists), §2 accepted (FI keeps "skill"), §4 downgraded from blocker to
convenience (origin is recoverable from the `proj_…::` id prefix), and §3
withdrawn entirely — FI pins a single project (`core/knowledge.py:189`), so a
single-project query already spans every quest and the many-project requirement
was premised on a layout FI does not have.

**Net: nothing remains that Axon must schedule on FI's behalf.** Two items were
handed over as Axon's own to weigh, not as requests:

1. **The `default`-prefix residual.** Id namespacing at `src/axon/main.py:2429-2441`
   deliberately exempts the `default` project, so `default`-versus-other
   collisions remain possible in `MultiVectorStore` dedup. Narrow, and it does
   not affect FI, which does not use `default`.

2. **`POST /add_text` returns 500** — `{"detail":"[Errno 22] Invalid argument"}`,
   reproduced with and without a `metadata` object and across two active
   projects. Re-checked 2026-09-06: still reproducing, **but against the same
   long-running server process**, which predates any fix. Restarting `axon-api`
   is required before that re-check means anything. Treat the original report as
   the evidence, not this one.

## If the fourth reply is a rebuttal of #3

The two claims in #3 most worth contesting, so they are easy to find:

- That §4 defect 2 is **already fixed** — resting on the comment at
  `src/axon/main.py:2429-2441` naming `MultiVectorStore` / `MultiBM25Retriever`
  dedup as the exact hazard it prevents, plus live ids of the form
  `proj_51d42033…::file_98c8ff97…_p5_chunk_2` observed from `search_knowledge`.
- That origin project is **recoverable without a `project` metadata key**, by
  mapping the id prefix through `get_project_id`.

If either is wrong, the ordering conclusion in #3 goes with it.
