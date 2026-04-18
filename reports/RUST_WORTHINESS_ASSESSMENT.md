# Rust Worthiness Assessment

Generated: 2026-04-18T13:38:15.622142+00:00

Measured candidates:

| Candidate | Baseline ms | Candidate ms | Peak RAM | Disk bytes | LLM calls | Accuracy | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| llm_fused_extraction | 67.2 | 35.555 | 204.0 KB | - | 24 | exact_match=True | keep |
| relation_graph_msgpack_persistence | 160.62 | 468.615 | 10.0 MB | 896.6 KB | 0 | exact_match=False | reject |
| alias_resolution_rust_grouping | 2947.771 | 1545.859 | 976.0 KB | - | 0 | entity_graph_match=True, relation_graph_match=True | keep |
| community_detection_rust | 1224.734 | 19.116 | 841.9 KB | - | 0 | same_nodes=True, community_count_delta=0 | keep |
| build_graph_edges_rust | 67.926 | 93.977 | 6.2 MB | - | 0 | same_node_set=True, same_edge_count=True | reject |
| merge_entities_into_graph_rust | 82.728 | 94.536 | 3.4 MB | - | 0 | entity_graph_match=False | reject |

Keep/reject summary:

- `llm_fused_extraction`: `keep`. Fused prompt cuts entity+relation LLM completions from 2 per chunk to 1.
- `relation_graph_msgpack_persistence`: `reject`. Msgpack persistence should reduce on-disk bytes and JSON encode/decode overhead.
- `alias_resolution_rust_grouping`: `keep`. Rust removes the dense NumPy similarity matrix but keeps Python-side canonical merge semantics.
- `community_detection_rust`: `keep`. This measures the actual `_run_community_detection` wiring with and without Rust Louvain.
- `build_graph_edges_rust`: `reject`. Benchmarks the actual `_build_graph_edge_payload` bridge branch.
- `merge_entities_into_graph_rust`: `reject`. Compares the inline Python merge loop against the wired Rust helper semantics.

Deferred candidates:

- `local_search_context`: not promoted. The hot path is dominated by vector-store fetches, Python string assembly, and context formatting; isolated Rust ranking would not move the end-to-end latency enough without also moving retrieval and formatting boundaries.
- `merge_relations_into_graph`: promising follow-up if relation batches get much larger. The Rust primitive exists, but this pass prioritized disk, RAM, and LLM-call reductions with clearer wins.
- `code_graph build/persist`: worth a separate study when code-ingest corpora exceed the current qualification sizes. The likely win is a codec plus batched edge construction, not a direct line-for-line port.
