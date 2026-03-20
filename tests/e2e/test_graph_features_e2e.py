from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.integration]


def test_graphrag_endpoints_build_payload_and_html(api_client, make_brain, monkeypatch):
    brain = make_brain(
        graph_rag=True,
        graph_rag_relations=True,
        graph_rag_community=False,
        graph_rag_community_defer=False,
        graph_rag_min_entities_for_relations=0,
    )

    def fake_extract_entities(text: str) -> list[dict]:
        entities = []
        lowered = text.lower()
        if "axon" in lowered:
            entities.append({"name": "Axon", "type": "PRODUCT", "description": "RAG platform"})
        if "bm25" in lowered:
            entities.append({"name": "BM25", "type": "CONCEPT", "description": "Keyword retrieval"})
        if "vector search" in lowered:
            entities.append(
                {"name": "Vector Search", "type": "CONCEPT", "description": "Semantic retrieval"}
            )
        return entities

    def fake_extract_relations(text: str) -> list[dict]:
        lowered = text.lower()
        relations = []
        if "axon uses bm25" in lowered:
            relations.append(
                {
                    "subject": "Axon",
                    "relation": "USES",
                    "object": "BM25",
                    "description": "Axon uses BM25",
                }
            )
        if "axon combines vector search" in lowered:
            relations.append(
                {
                    "subject": "Axon",
                    "relation": "COMBINES",
                    "object": "Vector Search",
                    "description": "Axon combines vector search",
                }
            )
        return relations

    monkeypatch.setattr(brain, "_extract_entities", fake_extract_entities)
    monkeypatch.setattr(brain, "_extract_relations", fake_extract_relations)

    batch = api_client.post(
        "/add_texts",
        json={
            "docs": [
                {
                    "doc_id": "graph_doc_a",
                    "text": "Axon uses BM25 for lexical retrieval.",
                    "metadata": {"source": "graph_a.txt"},
                },
                {
                    "doc_id": "graph_doc_b",
                    "text": "Axon combines vector search with grounded retrieval.",
                    "metadata": {"source": "graph_b.txt"},
                },
            ]
        },
    )
    assert batch.status_code == 200

    status = api_client.get("/graph/status")
    assert status.status_code == 200
    assert status.json()["community_build_in_progress"] is False

    graph_data = api_client.get("/graph/data")
    assert graph_data.status_code == 200
    payload = graph_data.json()
    node_ids = {node["id"] for node in payload["nodes"]}
    relations = {(link["source"], link["target"], link["relation"]) for link in payload["links"]}
    assert {"axon", "bm25", "vector search"}.issubset(node_ids)
    assert ("axon", "bm25", "USES") in relations

    visualize = api_client.get("/graph/visualize")
    assert visualize.status_code == 200
    assert "GraphRAG 3D Viewer" in visualize.text
    assert "Axon" in visualize.text

    finalize = api_client.post("/graph/finalize")
    assert finalize.status_code == 200
    assert finalize.json()["status"] == "ok"


def test_code_graph_endpoints_and_code_query_diagnostics(api_client, make_brain, tmp_path: Path):
    make_brain(code_graph=True)

    code_dir = tmp_path / "code"
    code_dir.mkdir(parents=True, exist_ok=True)
    (code_dir / "pipeline.py").write_text(
        "import os\n\n"
        "def build_graph(data):\n"
        "    return {'nodes': len(data), 'cwd': os.getcwd()}\n\n"
        "class Runner:\n"
        "    def execute(self, items):\n"
        "        return build_graph(items)\n",
        encoding="utf-8",
    )

    ingest = api_client.post("/ingest", json={"path": str(code_dir)})
    assert ingest.status_code == 200
    job_id = ingest.json()["job_id"]
    status = api_client.get(f"/ingest/status/{job_id}")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"

    code_graph = api_client.get("/code-graph/data")
    assert code_graph.status_code == 200
    payload = code_graph.json()
    node_types = {node["type"] for node in payload["nodes"]}
    edge_labels = {edge["label"] for edge in payload["links"]}
    assert "file" in node_types
    assert "function" in node_types or "method" in node_types
    assert "CONTAINS" in edge_labels

    raw_search = api_client.post(
        "/search/raw",
        json={"query": "Where is build_graph defined?", "top_k": 5},
    )
    assert raw_search.status_code == 200
    raw_payload = raw_search.json()
    assert raw_payload["results"]
    assert raw_payload["diagnostics"]["code_mode_triggered"] is True
    assert raw_payload["diagnostics"]["result_count"] >= 1
