from __future__ import annotations

import copy
import gc
import json
import os
import sys
import tempfile
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from axon.config import AxonConfig
from axon.graph_rag import GraphRagMixin
from axon.rust_bridge import get_rust_bridge

OUT_DIR = REPO_ROOT / "reports"
JSON_OUT = OUT_DIR / "rust_worthiness_assessment.json"
MD_OUT = OUT_DIR / "RUST_WORTHINESS_ASSESSMENT.md"


class NullBridge:
    def can_build_graph_edges(self):
        return False

    def can_run_louvain(self):
        return False

    def can_merge_entities_into_graph(self):
        return False

    def can_relation_graph_codec(self):
        return False

    def can_resolve_entity_alias_groups(self):
        return False


class FakeVectorStore:
    def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [{"id": doc_id, "text": f"text for {doc_id}", "metadata": {}} for doc_id in ids]


class FakeOwnVectorStore:
    def add(self, ids, texts, embeddings, metadatas) -> None:
        return None


class CountingLLM:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0
        self.prompt_chars = 0

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.calls += 1
        self.prompt_chars += len(prompt) + len(system_prompt or "")
        if self.delay_s:
            time.sleep(self.delay_s)
        if "exactly two top-level keys" in prompt:
            return json.dumps(
                {
                    "entities": [
                        {"name": "Alice", "type": "PERSON", "description": "Engineer"},
                        {"name": "Bob", "type": "PERSON", "description": "Colleague"},
                    ],
                    "relations": [
                        {
                            "subject": "Alice",
                            "relation": "works_with",
                            "object": "Bob",
                            "description": "They collaborate closely.",
                            "strength": 8,
                        }
                    ],
                }
            )
        if "Extract the key named entities" in prompt:
            return "Alice | PERSON | Engineer\nBob | PERSON | Colleague"
        if "Extract key relationships" in prompt:
            return "Alice | works_with | Bob | They collaborate closely. | 8"
        return ""


class DeterministicEmbedding:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self.vectors.get(query, [0.0, 0.0, 0.0])


def _make_brain(tmp_dir: Path, *, llm=None, embedding=None, **cfg_overrides):
    cfg = AxonConfig(
        bm25_path=str(tmp_dir / "bm25"),
        vector_store_path=str(tmp_dir / "vector"),
    )
    for key, value in {
        "graph_rag": True,
        "graph_rag_llm_cache": False,
        "graph_rag_extraction_cache": False,
        **cfg_overrides,
    }.items():
        setattr(cfg, key, value)
    os.makedirs(cfg.bm25_path, exist_ok=True)

    class BenchBrain(GraphRagMixin):
        pass

    brain = BenchBrain()
    brain.config = cfg
    brain._entity_graph = {}
    brain._relation_graph = {}
    brain._community_levels = {}
    brain._community_hierarchy = {}
    brain._community_children = {}
    brain._community_summaries = {}
    brain._entity_embeddings = {}
    brain._claims_graph = {}
    brain._graph_rag_cache = {}
    brain._graph_rag_cache_dirty = False
    brain._text_unit_relation_map = {}
    brain._entity_description_buffer = {}
    brain._relation_description_buffer = {}
    brain._community_graph_dirty = False
    brain._nx_graph = None
    brain._nx_graph_dirty = True
    brain._incoming_rel_sig = None
    brain.vector_store = FakeVectorStore()
    brain._own_vector_store = FakeOwnVectorStore()
    brain.llm = llm or CountingLLM()
    brain.embedding = embedding or DeterministicEmbedding({})
    brain._executor = ThreadPoolExecutor(max_workers=8)
    return brain


def _shutdown_brain(brain) -> None:
    executor = getattr(brain, "_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


def _rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _measure(fn):
    gc.collect()
    rss_before = _rss_bytes()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_bytes()
    rss_delta = None
    if rss_before is not None and rss_after is not None:
        rss_delta = max(0, rss_after - rss_before)
    peak_ram_bytes = int(max(peak, rss_delta or 0))
    return result, {"elapsed_ms": round(elapsed_ms, 3), "peak_ram_bytes": peak_ram_bytes}


def _artifact_bytes(root: Path, prefix: str) -> int:
    total = 0
    for path in root.glob(f"{prefix}*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _normalize_entity_graph(graph: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in graph.items():
        if not isinstance(value, dict):
            continue
        node = dict(value)
        node["chunk_ids"] = sorted(str(x) for x in node.get("chunk_ids", []))
        normalized[key] = node
    return normalized


def _normalize_relation_graph(graph: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for key, entries in graph.items():
        clean_entries = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            clean_entries.append(dict(sorted(entry.items())))
        normalized[key] = sorted(
            clean_entries,
            key=lambda item: (
                str(item.get("target", "")),
                str(item.get("relation", "")),
                str(item.get("chunk_id", "")),
                str(item.get("subject", "")),
                str(item.get("object", "")),
            ),
        )
    return normalized


def _python_merge_entities(entity_graph: dict[str, dict[str, Any]], results: list[tuple[str, list[dict]]]) -> int:
    inserted = 0
    for doc_id, entities in results:
        for ent in entities:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            key = ent["name"].lower().strip()
            if not key:
                continue
            if key not in entity_graph:
                entity_graph[key] = {
                    "description": ent.get("description", ""),
                    "type": ent.get("type", "UNKNOWN"),
                    "chunk_ids": [],
                    "frequency": 0,
                    "degree": 0,
                }
                inserted += 1
            node = entity_graph[key]
            if doc_id not in node["chunk_ids"]:
                node["chunk_ids"].append(doc_id)
            if not node.get("description") and ent.get("description"):
                node["description"] = ent["description"]
            if not node.get("type"):
                node["type"] = ent.get("type", "UNKNOWN")
            node["frequency"] = len(node["chunk_ids"])
            node.setdefault("degree", 0)
    return inserted


def _make_relation_dataset(entity_count: int = 2500, rels_per_entity: int = 3):
    entity_graph = {}
    relation_graph = {}
    for i in range(entity_count):
        entity_graph[f"entity_{i}"] = {
            "description": f"entity {i}",
            "chunk_ids": [f"c{i}"],
            "frequency": 1,
            "degree": 0,
            "type": "CONCEPT",
        }
        entries = []
        for j in range(1, rels_per_entity + 1):
            tgt = f"entity_{(i + j) % entity_count}"
            entries.append(
                {
                    "target": tgt,
                    "relation": "related_to",
                    "chunk_id": f"c{i}",
                    "subject": f"entity_{i}",
                    "object": tgt,
                    "weight": float(j),
                    "strength": min(10, 4 + j),
                }
            )
        relation_graph[f"entity_{i}"] = entries
    return entity_graph, relation_graph


def _make_community_dataset(cluster_count: int = 40, cluster_size: int = 40):
    entity_graph = {}
    relation_graph = {}
    total = cluster_count * cluster_size
    for i in range(total):
        entity_graph[f"node_{i}"] = {
            "description": f"Node {i}",
            "chunk_ids": [f"c{i}"],
            "frequency": 1,
            "degree": 0,
            "type": "CONCEPT",
        }
    for cluster in range(cluster_count):
        start = cluster * cluster_size
        for offset in range(cluster_size):
            src_idx = start + offset
            src = f"node_{src_idx}"
            entries = relation_graph.setdefault(src, [])
            for step in (1, 2, 3):
                tgt = f"node_{start + ((offset + step) % cluster_size)}"
                entries.append({"target": tgt, "relation": "cluster", "chunk_id": f"c{src_idx}", "weight": 2.0})
            if cluster < cluster_count - 1 and offset == 0:
                bridge_tgt = f"node_{start + cluster_size}"
                entries.append(
                    {
                        "target": bridge_tgt,
                        "relation": "bridge",
                        "chunk_id": f"c{src_idx}",
                        "weight": 0.25,
                    }
                )
    return entity_graph, relation_graph


def _make_alias_dataset(pair_count: int = 400):
    entity_graph = {}
    relation_graph = {}
    vectors: dict[str, list[float]] = {}
    dims = pair_count + 16
    for i in range(pair_count):
        base = f"company_{i}"
        alias = f"{base} ltd"
        vec = [0.0] * dims
        vec[i] = 1.0
        vec[(i + 7) % dims] += 0.2
        alias_vec = list(vec)
        alias_vec[i % dims] += 0.001
        entity_graph[base] = {
            "description": f"{base} canonical",
            "chunk_ids": [f"c{i}", f"c{i}_a"],
            "frequency": 2,
            "degree": 0,
            "type": "ORGANIZATION",
        }
        entity_graph[alias] = {
            "description": "",
            "chunk_ids": [f"c{i}_b"],
            "frequency": 1,
            "degree": 0,
            "type": "ORGANIZATION",
        }
        relation_graph[alias] = [
            {
                "target": "shared_partner",
                "relation": "partner",
                "chunk_id": f"c{i}_b",
                "subject": alias,
                "object": "shared_partner",
            }
        ]
        vectors[base] = vec
        vectors[alias] = alias_vec
    return entity_graph, relation_graph, vectors


def _make_merge_results(unique_entities: int = 1200, chunk_count: int = 3000, ents_per_chunk: int = 3):
    results: list[tuple[str, list[dict[str, Any]]]] = []
    for i in range(chunk_count):
        entities = []
        for offset in range(ents_per_chunk):
            idx = (i + offset * 17) % unique_entities
            entities.append(
                {
                    "name": f"Entity {idx}",
                    "type": "CONCEPT",
                    "description": f"Entity {idx} description",
                }
            )
        results.append((f"doc_{i}", entities))
    return results


def bench_llm_fused_extraction() -> dict[str, Any]:
    docs = [{"id": f"d{i}", "text": f"Alice collaborates with Bob on document {i}."} for i in range(24)]
    summary: dict[str, Any] = {"candidate": "llm_fused_extraction"}
    runs = {}
    outputs = {}
    for fused in (False, True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="axon_rust_llm_"))
        llm = CountingLLM(delay_s=0.01)
        brain = _make_brain(
            tmp_dir,
            llm=llm,
            graph_rag_llm_fused_extraction=fused,
            graph_rag_ner_backend="llm",
            graph_rag_relation_backend="llm",
            graph_rag_relation_budget=0,
            graph_rag_llm_cache=False,
            graph_rag_extraction_cache=False,
        )
        try:
            (result, stats) = _measure(
                lambda: brain._extract_graph_llm_batches(
                    docs,
                    relations_enabled=True,
                    min_entities_for_relations=0,
                    relation_budget=0,
                )
            )
            runs["fused" if fused else "baseline"] = {
                **stats,
                "llm_calls": llm.calls,
                "prompt_chars": llm.prompt_chars,
                "disk_bytes": 0,
            }
            outputs["fused" if fused else "baseline"] = result
        finally:
            _shutdown_brain(brain)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["fused"]
    summary["accuracy"] = {"exact_match": outputs["baseline"] == outputs["fused"]}
    summary["decision"] = (
        "keep"
        if runs["fused"]["llm_calls"] < runs["baseline"]["llm_calls"]
        and summary["accuracy"]["exact_match"]
        else "reject"
    )
    summary["notes"] = "Fused prompt cuts entity+relation LLM completions from 2 per chunk to 1."
    return summary


def bench_relation_graph_persistence() -> dict[str, Any]:
    bridge = get_rust_bridge()
    summary: dict[str, Any] = {"candidate": "relation_graph_msgpack_persistence"}
    if not bridge.can_relation_graph_codec():
        summary["status"] = "unsupported"
        summary["decision"] = "reject"
        summary["notes"] = "Rust relation-graph codec unavailable."
        return summary

    dataset = _make_relation_dataset()
    runs = {}
    loaded_graphs = {}
    for msgpack_mode in (False, True):
        tmp_dir = Path(tempfile.mkdtemp(prefix="axon_rust_relpersist_"))
        brain = _make_brain(
            tmp_dir,
            graph_rag_relation_msgpack_persist=msgpack_mode,
            graph_rag_relation_shard_persist=False,
        )
        brain._relation_graph = copy.deepcopy(dataset[1])
        try:
            (_, save_stats) = _measure(brain._save_relation_graph)
            (loaded, load_stats) = _measure(brain._load_relation_graph)
            label = "candidate" if msgpack_mode else "baseline"
            runs[label] = {
                "save_elapsed_ms": save_stats["elapsed_ms"],
                "load_elapsed_ms": load_stats["elapsed_ms"],
                "peak_ram_bytes": max(save_stats["peak_ram_bytes"], load_stats["peak_ram_bytes"]),
                "disk_bytes": _artifact_bytes(Path(brain.config.bm25_path), ".relation_graph"),
                "llm_calls": 0,
            }
            loaded_graphs[label] = loaded
        finally:
            _shutdown_brain(brain)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["candidate"]
    summary["accuracy"] = {
        "exact_match": _normalize_relation_graph(loaded_graphs["baseline"])
        == _normalize_relation_graph(loaded_graphs["candidate"])
    }
    summary["decision"] = (
        "keep"
        if runs["candidate"]["disk_bytes"] < runs["baseline"]["disk_bytes"]
        and summary["accuracy"]["exact_match"]
        else "reject"
    )
    summary["notes"] = "Msgpack persistence should reduce on-disk bytes and JSON encode/decode overhead."
    return summary


def bench_alias_resolution() -> dict[str, Any]:
    bridge = get_rust_bridge()
    summary: dict[str, Any] = {"candidate": "alias_resolution_rust_grouping"}
    if not bridge.can_resolve_entity_alias_groups():
        summary["status"] = "unsupported"
        summary["decision"] = "reject"
        summary["notes"] = "Rust alias-grouping helper unavailable."
        return summary

    entity_graph, relation_graph, vectors = _make_alias_dataset()
    embedding = DeterministicEmbedding(vectors)
    runs = {}
    outputs = {}
    for backend in ("numpy", "rust"):
        tmp_dir = Path(tempfile.mkdtemp(prefix="axon_rust_alias_"))
        brain = _make_brain(
            tmp_dir,
            embedding=embedding,
            graph_rag_entity_resolve_backend=backend,
            graph_rag_entity_resolve_threshold=0.995,
        )
        brain._entity_graph = copy.deepcopy(entity_graph)
        brain._relation_graph = copy.deepcopy(relation_graph)
        try:
            (merged, stats) = _measure(brain._resolve_entity_aliases)
            runs["candidate" if backend == "rust" else "baseline"] = {
                **stats,
                "merged": merged,
                "disk_bytes": 0,
                "llm_calls": 0,
            }
            outputs["candidate" if backend == "rust" else "baseline"] = (
                copy.deepcopy(brain._entity_graph),
                copy.deepcopy(brain._relation_graph),
            )
        finally:
            _shutdown_brain(brain)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["candidate"]
    summary["accuracy"] = {
        "entity_graph_match": _normalize_entity_graph(outputs["baseline"][0])
        == _normalize_entity_graph(outputs["candidate"][0]),
        "relation_graph_match": _normalize_relation_graph(outputs["baseline"][1])
        == _normalize_relation_graph(outputs["candidate"][1]),
    }
    summary["decision"] = (
        "keep"
        if runs["candidate"]["elapsed_ms"] < runs["baseline"]["elapsed_ms"]
        and all(summary["accuracy"].values())
        else "reject"
    )
    summary["notes"] = "Rust removes the dense NumPy similarity matrix but keeps Python-side canonical merge semantics."
    return summary


def bench_community_detection() -> dict[str, Any]:
    bridge = get_rust_bridge()
    summary: dict[str, Any] = {"candidate": "community_detection_rust"}
    if not bridge.can_run_louvain():
        summary["status"] = "unsupported"
        summary["decision"] = "reject"
        summary["notes"] = "Rust Louvain unavailable."
        return summary

    entity_graph, relation_graph = _make_community_dataset()
    runs = {}
    outputs = {}
    for label, bridge_patch in (
        ("baseline", NullBridge()),
        ("candidate", None),
    ):
        tmp_dir = Path(tempfile.mkdtemp(prefix="axon_rust_comm_"))
        brain = _make_brain(tmp_dir)
        brain._entity_graph = copy.deepcopy(entity_graph)
        brain._relation_graph = copy.deepcopy(relation_graph)
        try:
            with patch("axon.rust_bridge.get_rust_bridge", return_value=bridge_patch) if bridge_patch else _nullcontext():
                (mapping, stats) = _measure(brain._run_community_detection)
            runs[label] = {**stats, "disk_bytes": 0, "llm_calls": 0, "communities": len(set(mapping.values()))}
            outputs[label] = mapping
        finally:
            _shutdown_brain(brain)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["candidate"]
    summary["accuracy"] = {
        "same_nodes": set(outputs["baseline"].keys()) == set(outputs["candidate"].keys()),
        "community_count_delta": abs(runs["baseline"]["communities"] - runs["candidate"]["communities"]),
    }
    summary["decision"] = (
        "keep"
        if runs["candidate"]["elapsed_ms"] < runs["baseline"]["elapsed_ms"]
        and summary["accuracy"]["same_nodes"]
        else "reject"
    )
    summary["notes"] = "This measures the actual `_run_community_detection` wiring with and without Rust Louvain."
    return summary


def bench_build_graph_edges() -> dict[str, Any]:
    bridge = get_rust_bridge()
    summary: dict[str, Any] = {"candidate": "build_graph_edges_rust"}
    if not bridge.can_build_graph_edges():
        summary["status"] = "unsupported"
        summary["decision"] = "reject"
        summary["notes"] = "Rust graph-edge builder unavailable."
        return summary

    entity_graph, relation_graph = _make_relation_dataset(entity_count=5000, rels_per_entity=4)
    runs = {}
    outputs = {}
    for label, bridge_patch in (
        ("baseline", NullBridge()),
        ("candidate", None),
    ):
        tmp_dir = Path(tempfile.mkdtemp(prefix="axon_rust_edges_"))
        brain = _make_brain(tmp_dir, graph_rag_rust_build_edges=(label == "candidate"))
        brain._entity_graph = copy.deepcopy(entity_graph)
        brain._relation_graph = copy.deepcopy(relation_graph)
        try:
            with patch("axon.rust_bridge.get_rust_bridge", return_value=bridge_patch) if bridge_patch else _nullcontext():
                (payload, stats) = _measure(brain._build_graph_edge_payload)
            nodes, edges = payload
            runs[label] = {
                **stats,
                "nodes": len(nodes),
                "edges": len(edges),
                "disk_bytes": 0,
                "llm_calls": 0,
            }
            outputs[label] = (set(nodes), len(edges))
        finally:
            _shutdown_brain(brain)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["candidate"]
    summary["accuracy"] = {
        "same_node_set": outputs["baseline"][0] == outputs["candidate"][0],
        "same_edge_count": outputs["baseline"][1] == outputs["candidate"][1],
    }
    summary["decision"] = (
        "keep"
        if runs["candidate"]["elapsed_ms"] < runs["baseline"]["elapsed_ms"] * 0.9
        and all(summary["accuracy"].values())
        else "reject"
    )
    summary["notes"] = "Benchmarks the actual `_build_graph_edge_payload` bridge branch."
    return summary


def bench_merge_entities() -> dict[str, Any]:
    bridge = get_rust_bridge()
    summary: dict[str, Any] = {"candidate": "merge_entities_into_graph_rust"}
    if not bridge.can_merge_entities_into_graph():
        summary["status"] = "unsupported"
        summary["decision"] = "reject"
        summary["notes"] = "Rust entity-merge helper unavailable."
        return summary

    results = _make_merge_results()
    runs = {}
    outputs = {}
    for label in ("baseline", "candidate"):
        entity_graph: dict[str, dict[str, Any]] = {}
        if label == "baseline":
            (_, stats) = _measure(lambda: _python_merge_entities(entity_graph, copy.deepcopy(results)))
        else:
            (_, stats) = _measure(
                lambda: bridge.merge_entities_into_graph(entity_graph, copy.deepcopy(results))
            )
        runs[label] = {**stats, "nodes": len(entity_graph), "disk_bytes": 0, "llm_calls": 0}
        outputs[label] = copy.deepcopy(entity_graph)
    summary["baseline"] = runs["baseline"]
    summary["candidate_metrics"] = runs["candidate"]
    summary["accuracy"] = {
        "entity_graph_match": _normalize_entity_graph(outputs["baseline"])
        == _normalize_entity_graph(outputs["candidate"])
    }
    summary["decision"] = (
        "keep"
        if runs["candidate"]["elapsed_ms"] < runs["baseline"]["elapsed_ms"] * 0.9
        and summary["accuracy"]["entity_graph_match"]
        else "reject"
    )
    summary["notes"] = "Compares the inline Python merge loop against the wired Rust helper semantics."
    return summary


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def _format_bytes(value: int | None) -> str:
    if value in (None, 0):
        return "-"
    units = ["B", "KB", "MB", "GB"]
    size = float(value)
    unit = 0
    while size >= 1024.0 and unit < len(units) - 1:
        size /= 1024.0
        unit += 1
    return f"{size:.1f} {units[unit]}"


def _write_report(payload: dict[str, Any]) -> None:
    lines = [
        "# Rust Worthiness Assessment",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "Measured candidates:",
        "",
        "| Candidate | Baseline ms | Candidate ms | Peak RAM | Disk bytes | LLM calls | Accuracy | Decision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in payload["benchmarks"]:
        if item.get("status") == "unsupported":
            lines.append(
                f"| {item['candidate']} | - | - | - | - | - | unsupported | {item['decision']} |"
            )
            continue
        baseline = item["baseline"]
        candidate = item["candidate_metrics"]
        accuracy = ", ".join(f"{k}={v}" for k, v in item["accuracy"].items())
        lines.append(
            "| "
            f"{item['candidate']} | "
            f"{baseline.get('elapsed_ms', baseline.get('save_elapsed_ms', '-'))} | "
            f"{candidate.get('elapsed_ms', candidate.get('save_elapsed_ms', '-'))} | "
            f"{_format_bytes(candidate.get('peak_ram_bytes'))} | "
            f"{_format_bytes(candidate.get('disk_bytes'))} | "
            f"{candidate.get('llm_calls', '-')} | "
            f"{accuracy} | "
            f"{item['decision']} |"
        )
    lines.extend(
        [
            "",
            "Keep/reject summary:",
            "",
        ]
    )
    for item in payload["benchmarks"]:
        lines.append(f"- `{item['candidate']}`: `{item['decision']}`. {item.get('notes', '')}")
    lines.extend(
        [
            "",
            "Deferred candidates:",
            "",
            "- `local_search_context`: not promoted. The hot path is dominated by vector-store fetches, Python string assembly, and context formatting; isolated Rust ranking would not move the end-to-end latency enough without also moving retrieval and formatting boundaries.",
            "- `merge_relations_into_graph`: promising follow-up if relation batches get much larger. The Rust primitive exists, but this pass prioritized disk, RAM, and LLM-call reductions with clearer wins.",
            "- `code_graph build/persist`: worth a separate study when code-ingest corpora exceed the current qualification sizes. The likely win is a codec plus batched edge construction, not a direct line-for-line port.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks = [
        bench_llm_fused_extraction(),
        bench_relation_graph_persistence(),
        bench_alias_resolution(),
        bench_community_detection(),
        bench_build_graph_edges(),
        bench_merge_entities(),
    ]
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": benchmarks,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(payload)
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
