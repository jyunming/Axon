# GitHub Project Setup — Dynamic Graph Roadmap

Run these commands once to create and configure the GitHub Project board.
Replace `GITHUB_TOKEN` with a token that has `project` scope.

## Step 1 — Create the project

```bash
GITHUB_TOKEN="<token>" gh project create \
  --owner jyunming \
  --title "Axon Dynamic Graph Roadmap" \
  --format json
# Save the project number from output (e.g. 3) and the node ID (PVT_...)
```

## Step 2 — Add custom fields

```bash
# Phase field
GITHUB_TOKEN="<token>" gh api graphql -f query='
mutation {
  createProjectV2Field(input: {
    projectId: "<PROJECT_NODE_ID>"
    dataType: SINGLE_SELECT
    name: "Phase"
    singleSelectOptions: [
      {name: "M0 · Design Freeze",       color: GRAY,   description: "Internal — no version bump"},
      {name: "M1 · Regression Harness",  color: GRAY,   description: "Internal — no version bump"},
      {name: "M2 · Backend Boundary",    color: YELLOW, description: "Internal — no version bump"},
      {name: "v0.2 · Backend Projects",  color: GREEN,  description: "First user-visible release"},
      {name: "v0.3 · DynGraph Storage",  color: GREEN,  description: ""},
      {name: "v0.4 · DynGraph Queries",  color: GREEN,  description: ""},
      {name: "v1.0 · Hardening",         color: BLUE,   description: "Production release"}
    ]
  }) { projectV2Field { ... on ProjectV2SingleSelectField { id options { id name } } } }
}'

# Sprint field (12 weekly sprints from 2026-04-06)
GITHUB_TOKEN="<token>" gh api graphql -f query='
mutation {
  createProjectV2Field(input: {
    projectId: "<PROJECT_NODE_ID>"
    dataType: ITERATION
    name: "Sprint"
    iterationConfiguration: {
      startDate: "2026-04-06"
      duration: 7
      iterations: [
        {startDate: "2026-04-06",  title: "S1 · M0 Design Freeze",         duration: 7},
        {startDate: "2026-04-13",  title: "S2 · M1 Regression Harness",    duration: 7},
        {startDate: "2026-04-20",  title: "S3 · M2a Backend Factory",      duration: 7},
        {startDate: "2026-04-27",  title: "S4 · M2b Shim Removal",         duration: 7},
        {startDate: "2026-05-04",  title: "S5 · v0.2 Backend Projects",    duration: 7},
        {startDate: "2026-05-11",  title: "S6 · v0.3a SQLite Models",      duration: 7},
        {startDate: "2026-05-18",  title: "S7 · v0.3b Conflict/Lifecycle", duration: 7},
        {startDate: "2026-05-25",  title: "S8 · v0.4a Episode Ingest",     duration: 7},
        {startDate: "2026-06-01",  title: "S9 · v0.4b Temporal Query",     duration: 7},
        {startDate: "2026-06-08",  title: "S10 · v0.4c Federation",        duration: 7},
        {startDate: "2026-06-15",  title: "S11 · v1.0a Stress Tests",      duration: 7},
        {startDate: "2026-06-22",  title: "S12 · v1.0b Docs + Release",    duration: 7}
      ]
    }
  }) { projectV2Field { ... on ProjectV2IterationField {
    id configuration { iterations { id title startDate } }
  } } }
}'

# Due Date field
GITHUB_TOKEN="<token>" gh api graphql -f query='
mutation {
  createProjectV2Field(input: {
    projectId: "<PROJECT_NODE_ID>"
    dataType: DATE
    name: "Due Date"
  }) { projectV2Field { ... on ProjectV2Field { id } } }
}'

# Status field already exists by default as "Status" with Todo/In Progress/Done
```

## Step 3 — Seed M0 draft items

After getting the Phase field option IDs and Sprint iteration IDs from Step 2,
create draft items for Sprint 1:

```bash
# Create draft item
ITEM_ID=$(GITHUB_TOKEN="<token>" gh api graphql -f query='
  mutation {
    addProjectV2DraftIssue(input: {
      projectId: "<PROJECT_NODE_ID>"
      title: "Write ADR: backend immutability, mixed-backend limits, offline constraint"
    }) { projectItem { id } }
  }' --jq '.data.addProjectV2DraftIssue.projectItem.id')

# Set Phase, Sprint, Due Date on the item
GITHUB_TOKEN="<token>" gh api graphql -f query='
  mutation {
    updateProjectV2ItemFieldValue(input: {
      projectId: "<PROJECT_NODE_ID>"
      itemId: "'$ITEM_ID'"
      fieldId: "<PHASE_FIELD_ID>"
      value: { singleSelectOptionId: "<M0_OPTION_ID>" }
    }) { projectV2Item { id } }
  }'
```

## M0 draft items to create (Sprint 1, due 2026-04-12)

| Title | Phase |
|---|---|
| Write ADR: backend immutability, mixed-backend limits, offline constraint | M0 |
| Define `GraphBackend` protocol in `src/axon/graph_backends/base.py` (stubs only) | M0 |
| Document `GraphContext` and `GraphDataFilters` field lists | M0 |
| Document SQLite schema (episodes, entities, facts, fact_evidence) | M0 |
| Freeze conflict resolution policy and relation registry design | M0 |
| Document mixed-backend RRF federation rule | M0 |
| Document legacy project migration policy (lazy, graphrag default) | M0 |

## M1 draft items to create (Sprint 2, due 2026-04-19)

| Title | Phase |
|---|---|
| Create fixture: software_guide (FastAPI DI doc) | M1 |
| Create fixture: paper_abstract (RAG paper) | M1 |
| Create fixture: issue_thread (auth integration issue) | M1 |
| Create fixture: stdlib_docs (asyncio coroutines) | M1 |
| Create fixture: codebase (BM25Retriever module doc) | M1 |
| Create fixture: project_doc (CONTRIBUTING.md) | M1 |
| Write `tests/test_graphrag_parity.py` — ingest→extract→build→query→render | M1 |
| Write `tests/test_architecture.py` — shim removal enforcement (expected: failing) | M1 |

## Step 4 — Convert drafts to real issues when work starts

```bash
gh issue create \
  --title "M0: Write ADR — backend immutability and offline constraint" \
  --body "Part of the Dynamic Graph roadmap Phase 0 design freeze. See docs/DYNAMIC_GRAPH_ROADMAP.md." \
  --repo jyunming/Axon \
  --label "dynamic-graph,phase-0"

GITHUB_TOKEN="<token>" gh project item-add <PROJECT_NUMBER> \
  --owner jyunming \
  --url <issue-url>
```

## Notes

- Save field IDs and iteration IDs to `docs/sprint_process.md` after Step 2 —
  these are specific to this project instance and needed for all future item updates.
- The `GITHUB_TOKEN` needs `project` scope in addition to `repo` scope.
- Draft items do not appear in issue search; convert to real issues before
  assigning to engineers.
