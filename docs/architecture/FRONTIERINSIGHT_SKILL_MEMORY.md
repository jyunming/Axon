# What FrontierInsight will need from Axon

**Status:** proposal for discussion. Nothing here is agreed, scheduled, or
implemented on either side. Raised now because one requirement looks like it
touches Axon's retrieval contract, and that is cheaper to discuss before either
project builds against an assumption.

Written 2026-09-05, against Axon `4038b84`.

---

## 1. Context, briefly

FrontierInsight (FI) is moving from *generating* every experiment from scratch
to *accumulating* the ability to run them. The unit of accumulation is a
**skill**: what a piece of scientific software is for, how to drive it, an
executable self-test that proves it still works, and the range assertions its
outputs must satisfy.

Skills arrive four ways — learned by exploring unfamiliar software, distilled
from quests that succeeded, authored by hand, or generalised from a single
demonstration. A skill is only trusted once it passes a self-test it wrote and a
human approves it.

The part that concerns Axon is the second path. **Distillation is supposed to
mine the accumulated corpus of past projects, not merely the one quest that just
finished.** That is what makes the platform compound rather than merely
remember. FI's plan puts executable code on its own filesystem and everything
else — procedural knowledge, provenance, cross-project history — in Axon,
because Axon is already FI's cross-quest memory layer.

That plan assumes things about Axon that are worth checking with you.

---

## 2. A naming collision, first

`docs/SKILLS.md` uses **skill** to mean *an MCP tool Axon exposes to an agent* —
`search_knowledge`, `ingest_path`, and so on. FI is about to use **skill** to
mean *a learned procedure for driving third-party scientific software*.

Same word, unrelated referents, in two projects that talk to each other daily.
Worth settling before either set of docs hardens. FI can rename its concept
(candidates: *capability*, *procedure*, *instrument*) — Axon's usage is older
and already published, so the burden is reasonably on FI. Flagging it rather
than deciding it unilaterally.

---

## 3. Requirement — retrieval across several projects in one call

**This is the only requirement that looks like it needs API surface.**

Today `search_knowledge` and `query_knowledge` take `project: str | None`, i.e.
one project or the active one (`src/axon/mcp_server.py:188`, `:216`). Mounts
(`docs/AXON_STORE.md`) solve a different problem — read-only cross-*user*
sharing — not "query N of my own projects together".

FI's distillation step needs to ask questions of the form *"across every quest
that used this software, what actually worked?"* Each quest is naturally its own
project. So the query is inherently many-project.

Three shapes, roughly in order of how much they ask of Axon:

1. **FI fans out client-side** — N calls, one per project, merge and re-rank in
   FI. Needs nothing from Axon. Costs N embedding calls per question and puts
   the ranking quality in FI's hands, where it will be worse than yours.
2. **`project` accepts a list** — `project: str | list[str] | None`, results
   merged and ranked server-side, each carrying its origin. Smallest change we
   can see that solves it properly; presumably touches the retriever's
   collection selection rather than the API shape alone.
3. **A first-class project group** — a named, persistent set of projects that
   behaves as one retrieval namespace. Most useful to FI, largest ask, and it
   may overlap with whatever `children` in `list_projects` is intended to grow
   into.

FI can ship with (1) and would rather not. The question for the Axon team is
whether (2) is a small change or a deceptively large one — we can't tell from
outside, and it would shape FI's design either way.

---

## 4. Requirement — provenance that survives retrieval

A distilled skill needs to answer "which projects taught me this?" — both for
the human approval step, where you cannot approve what you cannot trace, and
because a skill derived from one lucky quest should carry less weight than one
derived from six.

Concretely, FI needs a retrieved chunk to reliably carry back the project it
came from and the ingest metadata it was stored with. If that is already true of
`search_knowledge` results today, this requirement disappears — it is listed
because we have not verified it, not because we believe it is missing.

---

## 5. Requirement — skill knowledge is not stale knowledge

`get_stale_docs` exists, and staleness is the right default for most ingested
material. Skill knowledge is the opposite case: a skill for software that has
not changed in two years is not stale, it is *settled*. Its self-test, not its
age, is the evidence.

FI needs a way to mark documents as exempt from staleness sweeps, or at least to
distinguish them, so that no retention job silently removes the substrate the
skill library is built on. A metadata convention may be enough; a class of
document that maintenance treats differently would be better.

---

## 6. Explicit non-requirements

To bound the ask:

- **No executable code in Axon.** Skill code, self-tests and entry points live
  on FI's filesystem. Axon stores knowledge and provenance only.
- **No sandbox or execution surface.** FI has its own venv and Docker executors.
- **No approval workflow.** Promotion is FI's gate, exercised through FI's own
  interfaces.
- **No new embedding or graph backend.** FI expects to use what Axon already has.

---

## 7. What we would like back

1. Is many-project retrieval (§3) something Axon wants, independent of FI? If it
   is only ever an FI need, client-side fan-out is the honest answer and we will
   take that path.
2. Do retrieved chunks already carry origin project + ingest metadata (§4)?
3. Is there an existing convention for documents that maintenance should leave
   alone (§5)?
4. Does the naming collision (§2) matter to you, or is FI free to keep the word?

No timeline attached. FI's own prerequisites — a numeric verification gate and a
prompt-template reordering — come first, and the skill machinery is several
steps behind them. The intent is to surface the coupling early rather than
arrive later with it already load-bearing.

---

## 8. Background

Fuller reasoning lives in FI's repo, not duplicated here:

- `dev/roadmap-ai-scientist-skills.md` — the clarified roadmap this derives from.
- `dev/research-accuracy-tools-tokens.md` — the audit behind it. Note its
  "trusted kernel" recommendation is superseded by the skills direction; the
  correctness and token findings still stand.
