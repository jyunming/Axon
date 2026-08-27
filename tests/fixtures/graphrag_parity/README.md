# GraphRAG Parity Fixtures

Used by `tests/test_graphrag_parity.py` (Phase 1 of the Dynamic Graph roadmap).

## Purpose

These fixtures let us verify GraphRAG behavior end-to-end — ingest → entity
extraction → relation extraction → community build → query → render payload —
**before** the Phase 2 backend boundary refactor starts.

The parity suite is about **refactor drift detection**, not extraction model
quality. All LLM calls are mocked with canned responses at the
`self.llm.complete()` boundary. GLiNER/REBEL paths are tested separately
under `@pytest.mark.slow`.

## Structure

Each fixture directory contains:

```
<scenario>/
  input.txt                 Source document text fed to ingest
  canned_extraction.json    Pre-defined LLM responses at the extraction boundary
  expected_graph.json       Expected entity/relation graph state after ingest
```

### `canned_extraction.json` format

Matches the pipe-delimited format that `_extract_entities` and
`_extract_relations` parse from `self.llm.complete()` responses.

```json
{
  "entities": "EntityName | ENTITY_TYPE | one-sentence description\n...",
  "relations": "SUBJECT | RELATION | OBJECT | one-sentence description | strength\n..."
}
```

`ENTITY_TYPE` must be one of: `PERSON`, `ORGANIZATION`, `GEO`, `EVENT`,
`CONCEPT`, `PRODUCT`.

`strength` is an integer 1–10 (1=weak, 10=core/defining).

### `expected_graph.json` format

The fixture files are the source of truth for this schema — it's looser
than a first-draft sketch might suggest, because entity/relation extraction
isn't fully deterministic and exact-match assertions on every field are
brittle:

```json
{
  "_comment": "optional human-readable note",
  "expected_entity_keys": ["EntityName", "..."],
  "expected_entity_types": {"EntityName": "PRODUCT"},
  "expected_relation_pairs": [["Subject", "Object"]]
}
```

`expected_entity_keys` lists every entity name the canned extraction
should produce (matched case-insensitively against `_entity_graph`'s
lowercased keys). `expected_entity_types` only needs to cover the subset
worth pinning a type for — not every entity. `expected_relation_pairs` is
`[subject, object]` pairs only (no relation label/strength assertions —
those aren't stable enough across extraction runs to pin).

## Fixture Scenarios

| Directory | Document type | Key entities | Why included |
|---|---|---|---|
| `software_guide/` | API tutorial doc | FastAPI, Starlette, Pydantic, Depends | Tests framework/library entity family |
| `paper_abstract/` | Academic paper abstract | RAG, LLM, Lewis et al., Wikipedia | Tests author/method/concept family |
| `issue_thread/` | GitHub issue discussion | Keycloak, OAuth2, FastAPI, JWT | Tests person/component/protocol family |
| `stdlib_docs/` | Python stdlib reference | Coroutine, Task, Future, asyncio | Tests CONCEPT-heavy, few relations |
| `codebase/` | Python source module | BM25Retriever, BM25Okapi, corpus | Tests code entity family (class/dep/method) |
| `project_doc/` | CONTRIBUTING.md guide | Axon, pytest, black, ruff, Git | Tests tool/process entity family |

## How the mock works

In `test_graphrag_parity.py`, `self.llm.complete` is patched with a side
effect keyed on the **prompt body**, not `system_prompt` — the real
prompts (`graph_rag.py`'s `_extract_entities`/`_extract_relations`) put
their instruction text in the user prompt, and `system_prompt` is just a
short role label (`"You are a named entity extraction specialist."`) that
never contains the substrings a naive matcher might reach for:

```python
def _canned_llm(prompt, system_prompt=None, **kwargs):
    prompt_l = prompt.lower()
    if "extract the key named entities" in prompt_l:
        return fixture["entities"]
    if "extract key relationships" in prompt_l:
        return fixture["relations"]
    return "no-op summary"  # community-summary prompts etc. — falls back
                             # gracefully to plain text, doesn't raise
```

This intercepts exactly the two `llm.complete()` calls made by
`_extract_entities` and `_extract_relations`, leaving all parsing,
graph-building, and community-detection logic running on real code. Test
config must also set `graph_rag_llm_fused_extraction=False` — otherwise
`_extract_graph_llm_batches` routes through a different, JSON-based
combined-extraction prompt this mock doesn't cover.
