# GitHub Project — Field IDs and Iteration IDs

Project: **Axon Dynamic Graph Roadmap**
URL: https://github.com/users/jyunming/projects/4
Project node ID: `PVT_kwHOBCfwF84BT2HN`

---

## Custom fields

| Field | Type | Node ID |
|---|---|---|
| Phase | Single-select | `PVTSSF_lAHOBCfwF84BT2HNzhBCMwI` |
| Sprint | Iteration | `PVTIF_lAHOBCfwF84BT2HNzhBCMzw` |
| Due Date | Date | `PVTF_lAHOBCfwF84BT2HNzhBCM2s` |

## Phase option IDs

| Phase | Option ID |
|---|---|
| M0 · Design Freeze | `b67ced46` |
| M1 · Regression Harness | `78ac5bc4` |
| M2 · Backend Boundary | `bcd687e4` |
| v0.2 · Backend Projects | `da3d5c42` |
| v0.3 · DynGraph Storage | `86351376` |
| v0.4 · DynGraph Queries | `35003f3a` |
| v1.0 · Hardening | `5b02a8d2` |

## Sprint iteration IDs

| Sprint | Iteration ID | Dates |
|---|---|---|
| S1 · M0 Design Freeze | `d201d9ca` | Apr 06–12 |
| S2 · M1 Regression Harness | `b7b4d73b` | Apr 13–19 |
| S3 · M2a Backend Factory | `7313a7ec` | Apr 20–26 |
| S4 · M2b Shim Removal | `a4170620` | Apr 27–May 03 |
| S5 · v0.2 Backend Projects | `b6a91569` | May 04–10 |
| S6 · v0.3a SQLite Models | `d0989775` | May 11–17 |
| S7 · v0.3b Conflict/Lifecycle | `6acb970e` | May 18–24 |
| S8 · v0.4a Episode Ingest | `3c37d0dc` | May 25–31 |
| S9 · v0.4b Temporal Query | `54b49fc1` | Jun 01–07 |
| S10 · v0.4c Federation | `b94631de` | Jun 08–14 |
| S11 · v1.0a Stress Tests | `f96606b3` | Jun 15–21 |
| S12 · v1.0b Docs + Release | `6907365c` | Jun 22–28 |

---

## Adding a new item

```bash
unset GITHUB_TOKEN

# 1. Create draft
ITEM_ID=$(gh api graphql -f query='
  mutation {
    addProjectV2DraftIssue(input: {
      projectId: "PVT_kwHOBCfwF84BT2HN"
      title: "Your item title"
    }) { projectItem { id } }
  }' --jq '.data.addProjectV2DraftIssue.projectItem.id')

# 2. Set Phase (replace OPTION_ID with value from table above)
gh api graphql -f query='
  mutation { updateProjectV2ItemFieldValue(input: {
    projectId: "PVT_kwHOBCfwF84BT2HN" itemId: "'$ITEM_ID'"
    fieldId: "PVTSSF_lAHOBCfwF84BT2HNzhBCMwI"
    value: { singleSelectOptionId: "OPTION_ID" }
  }) { projectV2Item { id } } }'

# 3. Set Sprint (replace ITERATION_ID with value from table above)
gh api graphql -f query='
  mutation { updateProjectV2ItemFieldValue(input: {
    projectId: "PVT_kwHOBCfwF84BT2HN" itemId: "'$ITEM_ID'"
    fieldId: "PVTIF_lAHOBCfwF84BT2HNzhBCMzw"
    value: { iterationId: "ITERATION_ID" }
  }) { projectV2Item { id } } }'

# 4. Set Due Date
gh api graphql -f query='
  mutation { updateProjectV2ItemFieldValue(input: {
    projectId: "PVT_kwHOBCfwF84BT2HN" itemId: "'$ITEM_ID'"
    fieldId: "PVTF_lAHOBCfwF84BT2HNzhBCM2s"
    value: { date: "2026-04-12" }
  }) { projectV2Item { id } } }'
```

## Converting a draft to a real issue

```bash
gh issue create \
  --title "M0: Write ADR — backend immutability" \
  --body "See docs/DYNAMIC_GRAPH_ROADMAP.md for context." \
  --repo jyunming/Axon \
  --label "dynamic-graph,phase-m0"

gh project item-add 4 --owner jyunming --url <issue-url>
```
