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

```json
{
  "entities": {
    "EntityName": {"type": "PRODUCT", "description": "..."}
  },
  "relations": [
    {"subject": "A", "relation": "built_on", "object": "B", "strength": 9}
  ]
}
```

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

In `test_graphrag_parity.py`, `self.llm.complete` is patched with a side effect
that looks up `canned_extraction.json` based on the prompt prefix:

```python
def _canned_llm(prompt, system_prompt="", **kwargs):
    if "named entities" in system_prompt:
        return fixture["entities"]
    if "relationships" in system_prompt:
        return fixture["relations"]
    return ""
```

This intercepts exactly the two `llm.complete()` calls made by
`_extract_entities` and `_extract_relations`, leaving all parsing,
graph-building, and community-detection logic running on real code.
