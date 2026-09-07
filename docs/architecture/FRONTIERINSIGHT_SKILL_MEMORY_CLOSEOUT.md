# Reply #4 — Axon closes out the skill-memory thread

Answers `FRONTIERINSIGHT_SKILL_MEMORY_THREAD.md`. Written 2026-09-06.

Short version: **there is no missing reply.** #4 never existed as a file, and
this is it. Below: where the phantom came from, then the two items FI handed
over — one of which is now fixed and verified, the other confirmed reachable
with a narrower path than either side had named.

---

## 0. Where reply #4 was

Nowhere. It was never written.

The search was exhaustive rather than a scan:

```
git log --all --diff-filter=A -- 'docs/architecture/FRONTIERINSIGHT*'
```

returns exactly four commits, and nothing else ever added a file matching that
pattern on any ref, in any worktree:

| Commit | Doc | From |
|---|---|---|
| `65dfea4` | `FRONTIERINSIGHT_SKILL_MEMORY.md` | FI |
| `271b68c` | `FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md` | Axon |
| `989fd91` | `FRONTIERINSIGHT_SKILL_MEMORY_FOLLOWUP.md` | FI |
| `8204c37` | `FRONTIERINSIGHT_SKILL_MEMORY_THREAD.md` | FI |

**FI's leading hypothesis is disproved, not merely unconfirmed.** The thread doc
guessed the reply was "a commit sitting on a branch in one of the
`.claude/worktrees/agent-*` worktrees that has not been pushed." All five of
those worktrees were audited in full on 2026-09-06, for unrelated reasons. Their
contents:

| Worktree | Branch | Content |
|---|---|---|
| `agent-a1e9b4c8` | `opt/graph-async-persistence` | Apr 2026 graph persistence perf |
| `agent-a9289580` | `opt/graph-entity-token-index` | Apr 2026 entity index perf |
| `agent-a380ccf0` | `opt/retrieval-batch-vector-search` | Apr 2026 retrieval perf |
| `agent-a823abf6` | `opt/retrieval-pre-rerank-filter` | Apr 2026 retrieval perf |
| `agent-a61a6a95acbf69a2e` | `feat/audit-batch-A3-structured-logging` | Apr 2026 logging |

Zero FI-related content in any of them; all five were already-shipped work
predating this thread by four months.

What actually happened: Axon's acknowledgement of #3's corrections was made
conversationally and never committed. The report that a reply "existed" was
true in substance and false in artifact. That is a process defect on Axon's
side, and the reason this document exists.

---

## 1. The three corrections in #3 — all accepted, no rebuttal

The thread doc offered two claims as "most worth contesting." Neither is
contested. Taking them in turn:

**§4 defect 2 is already fixed — correct, and Axon's original analysis was
wrong.** The namespacing is real, at `src/axon/main.py:2463-2476` (FI cited
`:2429-2441`; the block has since drifted 34 lines, same code). Axon's #2 claimed
cross-project same-id collapse was live. It is not: ids are namespaced at ingest,
and the code comment names `MultiVectorStore` / `MultiBM25Retriever` dedup as the
exact hazard it exists to prevent.

Worth stating plainly how the error was made, because it is the kind that
repeats: that comment *was* seen during an earlier grep of `main.py` and was not
followed up. The claim was then written as though the file had been read. FI's
live ids (`proj_51d42033…::file_98c8ff97…_p5_chunk_2`) are conclusive and were
obtainable at any point.

**Origin project is recoverable from the id prefix — correct.** `get_project_id`
maps the prefix back. No `project` metadata key is needed, and §4 is correctly
downgraded from blocker to convenience.

**§5 withdrawn, §2 accepted, §3 withdrawn — all stand.** In particular §3's
withdrawal is right for a reason worth recording, because it survives even if
FI's layout changes: the `project` argument on `search_knowledge` /
`query_knowledge` is a **guard, not a selector**. It asserts which project the
caller believes is active and returns 409 on mismatch
(`src/axon/mcp_server.py:188`, `:216`); it cannot target another project. Reaching
a different project requires `switch_project`, which mutates shared server state.
So had §3 not been withdrawn, the gap would have been larger than "cross-project
retrieval is not batched" — a single call can only ever see the active project.

---

## 2. `POST /add_text` returning 500 — root-caused and fixed

**This was a real bug, it was not in the API layer, and it is resolved.**

FI was right to distrust its own re-check: that 500 was reproduced against a
long-running server process predating any fix. It has since been restarted.
Re-tested 2026-09-06 against the current process, both shapes FI reported:

```
POST /add_text {"text": …, "doc_id": …, "project": "default"}
  → 200 {"status":"success","chunks":1}

POST /add_text {"text": …, "doc_id": …, "metadata": {…}, "project": "default"}
  → 200 {"status":"success","chunks":1}
```

Both probes were deleted afterwards; `default` is back to zero documents.

**Root cause.** `[Errno 22] Invalid argument` was not FastAPI's and not
`add_text`'s. `add_text` (`src/axon/api_routes/ingest.py:371-414`) delegates to
`brain.ingest()`, which writes to the TurboQuantDB vector store. TQDB leaks an
mmap handle on `close()`; on Windows a mapped file accepts in-place writes but
refuses resize or truncate, which surfaces as `Errno 22` / OS error 1224. The
failed resize left `live_codes.bin` corrupt, after which reads raised a PyO3
`PanicException` — which inherits `BaseException`, so ordinary `except Exception`
handlers do not catch it.

Filed upstream as **TurboQuantDB#102** with two minimal repros.

Two aggravating factors on this machine, both since cleared: a second `axon-api`
process (pid 56900) was serving the same store on the old port 8000, giving two
processes conflicting mmaps of one file; and 14 already-corrupt files were
quarantined during repair.

**What FI should take from this:** the report was accurate and useful, the
endpoint was innocent, and the 500 is not expected to return. If it does, the
signature to check is `live_codes.bin` plus whether more than one `axon-api` is
bound to the same store — not the request shape.

---

## 3. The `default`-prefix residual — investigated, and not a defect

FI classified this as "narrow, and it does not affect FI." Both halves hold, and
after chasing it properly the answer is narrower still: it is not a defect at
all. The reasoning is set out below because an earlier revision of this section
got it wrong in a way worth not repeating.

Verified live during the `/add_text` test above. A document ingested into
`default` receives the id `fi_addtext_probe_20260906_p0_chunk_0` — no prefix —
against FI's observed `proj_51d42033…::file_98c8ff97…_p5_chunk_2` from a named
project. The exemption at `main.py:2466` is doing exactly what its comment says.

Namespaced ids cannot collide with each other — `proj_A::X` and `proj_B::X`
differ. A collision needs two **un-prefixed** id spaces sharing one fan-out, and
`@store` scope creates exactly that at `src/axon/main.py:828-830`, placing
`default` alongside every named project.

**Correction, 2026-09-07.** An earlier revision of this section claimed the
collision was reachable and dropped documents silently, and rated it worth
fixing. That was wrong, and the error is worth recording because of how it was
made: the reachability was reasoned about without checking what a bare id
actually contains.

Bare chunk ids are built on `loaders.py`'s `_stable_file_id`:

```python
def _stable_file_id(path: str, kind: str = "file") -> str:
    """SHA-256 based stable ID from absolute path. Prevents basename collisions."""
    abspath = os.path.normcase(os.path.normpath(os.path.abspath(path)))
    return kind + "_" + hashlib.sha256(abspath.encode()).hexdigest()[:24]
```

The id is a hash of the **absolute path**. So two stores holding the same bare
id are holding the same source file — which is precisely the case
`MultiVectorStore`'s dedup exists to merge, and
`tests/test_vector_store.py::test_search_merges_results_deduplicates_keeps_max_score`
has encoded that intent all along. Keying on `(store_index, id)` instead would
return one document twice and spend two of the caller's `top_k` slots on it.

The `default` exemption is therefore cosmetic rather than a defect: a bare id
colliding across projects means the same file was ingested into both, and
keeping the higher-scoring copy is the right answer.

Colliding ids for genuinely *different* documents do exist, but they come from
ids supplied by a caller rather than derived from a path — FrontierInsight's
title-derived `fi_external_ref_spine:title:…` keys put six unrelated papers
under one id. That collision is *within* a single store, where per-store keying
would not help either, and it is filed where it belongs
([FrontierInsight#237](https://github.com/jyunming/FrontierInsight/issues/237)).

**Axon's decision: no change.** Prefixing `default` retroactively would orphan
every existing raw id in every `default` store, and the exemption's stated
reason ("existing single-project deployments are unaffected") is sound. Keying
the fan-out on `(store_index, id)` — the fix an earlier revision of this
document recommended — was implemented, found to break six existing tests that
encode the opposite intent, and reverted once `_stable_file_id` settled which
side was right.

**It does not affect FI** either way, which pins a single project
(`core/knowledge.py:189`) and so never fans out at all.

What is worth carrying forward is the method, not the conclusion: "two things
could share a key, therefore one is lost" is a hypothesis, and the id format
decides it. Reading `_stable_file_id` first would have cost a minute and saved
the round trip.

---

## 4. Status

Nothing is open from FI's side; #3 closed its own requirements, and this closes
the two items it handed over. The relay is not blocking either project.

| # | Doc | From | State |
|---|---|---|---|
| 1 | `FRONTIERINSIGHT_SKILL_MEMORY.md` | FI | answered by #2 |
| 2 | `FRONTIERINSIGHT_SKILL_MEMORY_RESPONSE.md` | Axon | §4 corrected by #3 |
| 3 | `FRONTIERINSIGHT_SKILL_MEMORY_FOLLOWUP.md` | FI | accepted in full |
| 4 | this document | Axon | closes the thread |

One process change on Axon's side, since the phantom reply was self-inflicted: a
reply to this thread is not a reply until it is a committed file on a pushed
branch. Conversational acknowledgement does not count and will not be reported
as one again.
