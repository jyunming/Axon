"""
Demo: Entity distance in Axon knowledge graph.

Seeds a toy entity/relation graph directly into a running Axon API server,
then calls GET /graph/distances to show hop distances from a seed entity.

Usage:
    python demo_entity_distance.py
"""
import json
import sys

import networkx as nx
import requests

BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# 1. Build the toy graph in-memory (mirrors what GraphRagMixin stores)
# ---------------------------------------------------------------------------

ENTITIES = {
    "alice": {"type": "PERSON", "description": "Lead engineer at Acme", "chunk_ids": ["c1"]},
    "acme corp": {"type": "ORGANIZATION", "description": "Technology company", "chunk_ids": ["c2"]},
    "bigcorp": {
        "type": "ORGANIZATION",
        "description": "Parent holding company",
        "chunk_ids": ["c3"],
    },
    "robert smith": {
        "type": "PERSON",
        "description": "CEO and founder of BigCorp",
        "chunk_ids": ["c4"],
    },
    "techventures": {
        "type": "ORGANIZATION",
        "description": "VC firm run by Robert",
        "chunk_ids": ["c5"],
    },
    "bob": {"type": "PERSON", "description": "Intern at Acme", "chunk_ids": ["c6"]},
}

RELATIONS = {
    "alice": [{"target": "acme corp", "relation": "WORKS_FOR", "weight": 8.0}],
    "acme corp": [
        {"target": "bigcorp", "relation": "SUBSIDIARY_OF", "weight": 6.0},
        {"target": "bob", "relation": "EMPLOYS", "weight": 4.0},
    ],
    "bigcorp": [{"target": "robert smith", "relation": "FOUNDED_BY", "weight": 9.0}],
    "robert smith": [{"target": "techventures", "relation": "CEO_OF", "weight": 7.0}],
}


def build_nx_graph():
    G = nx.DiGraph()
    for name, data in ENTITIES.items():
        G.add_node(name, **data)
    for src, rels in RELATIONS.items():
        for r in rels:
            w = r["weight"]
            G.add_edge(
                src,
                r["target"],
                distance=round(1.0 / (w + 1e-6), 4),
                relation=r["relation"],
                weight=w,
            )
    return G


# ---------------------------------------------------------------------------
# 2. Compute distances with Dijkstra (mirrors the /graph/distances endpoint)
# ---------------------------------------------------------------------------


def compute_distances(G, seed, max_hops=3):
    path_map = nx.single_source_shortest_path(G, seed, cutoff=max_hops)
    dist_map, _ = nx.single_source_dijkstra(G, seed, weight="distance")
    result = {}
    for entity, path_nodes in path_map.items():
        hop_count = len(path_nodes) - 1
        dist = dist_map.get(entity, float(hop_count))
        path_triples = [
            (
                path_nodes[i],
                G.edges[path_nodes[i], path_nodes[i + 1]].get("relation", ""),
                path_nodes[i + 1],
            )
            for i in range(len(path_nodes) - 1)
        ]
        result[entity] = {"hop_count": hop_count, "distance": round(dist, 4), "path": path_triples}
    return result


# ---------------------------------------------------------------------------
# 3. ASCII visualization
# ---------------------------------------------------------------------------

HOP_COLOR = {0: "\033[97m", 1: "\033[96m", 2: "\033[94m", 3: "\033[34m"}
RESET = "\033[0m"
GREY = "\033[90m"


def ascii_graph(distances, seed, all_entities):
    print(f"\n{'='*62}")
    print(f"  Entity distances from: '{seed}'  (max_hops=3)")
    print(f"{'='*62}")
    print(f"  {'Entity':<20} {'Hop':>4}  {'Dist':>6}  Path")
    print(f"  {'-'*20} {'-'*4}  {'-'*6}  {'-'*26}")

    # Sort by hop_count then entity name
    sorted_entities = sorted(
        all_entities, key=lambda e: (distances.get(e, {}).get("hop_count", 99), e)
    )

    for entity in sorted_entities:
        if entity not in distances:
            print(f"  {GREY}{entity:<20} {'—':>4}  {'—':>6}  (unreachable){RESET}")
            continue
        d = distances[entity]
        hop = d["hop_count"]
        color = HOP_COLOR.get(hop, GREY)
        if d["path"]:
            # Walk triples: build "A -[R]-> B -[R2]-> C"
            parts = [d["path"][0][0]]
            for s, r, o in d["path"]:
                parts.append(f"-[{r}]->")
                parts.append(o)
            path_str = " ".join(parts)
        else:
            path_str = "(seed)"
        print(f"  {color}{entity:<20}{RESET} {hop:>4}  {d['distance']:>6.4f}  {path_str}")

    print(
        f"\n  Legend: {HOP_COLOR[0]}[HOP 0 seed]{RESET}  "
        f"{HOP_COLOR[1]}[HOP 1]{RESET}  "
        f"{HOP_COLOR[2]}[HOP 2]{RESET}  "
        f"{HOP_COLOR[3]}[HOP 3]{RESET}  "
        f"{GREY}[unreachable]{RESET}"
    )
    print()


def graph_topology():
    print("\n  Toy knowledge graph topology:")
    print()
    print(
        "  alice  --[WORKS_FOR w=8]-->  acme corp  --[SUBSIDIARY_OF w=6]-->  bigcorp  --[FOUNDED_BY w=9]-->  robert smith  --[CEO_OF w=7]-->  techventures"
    )
    print("                                    +--[EMPLOYS w=4]--> bob")
    print()
    print(
        "  Edge distance = 1 / (weight + eps)  =>  heavier edge = shorter distance = preferred Dijkstra path"
    )
    print()


# ---------------------------------------------------------------------------
# 4. Optionally hit the live /graph/distances endpoint
# ---------------------------------------------------------------------------


def try_live_endpoint(seed, max_hops=3):
    try:
        resp = requests.get(
            f"{BASE}/graph/distances", params={"from_entity": seed, "max_hops": max_hops}, timeout=5
        )
        if resp.status_code == 200:
            print(f"  [live /graph/distances] status=200  entities returned: {len(resp.json())}")
        elif resp.status_code == 404:
            print(
                "  [live /graph/distances] 404 — entity not in server's graph (expected if LLM extraction didn't run)"
            )
        else:
            print(f"  [live /graph/distances] status={resp.status_code}")
    except Exception as e:
        print(f"  [live /graph/distances] server unreachable: {e}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "alice"
    max_hops = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    G = build_nx_graph()
    distances = compute_distances(G, seed, max_hops)

    graph_topology()
    ascii_graph(distances, seed, list(ENTITIES.keys()))

    print("  Raw JSON output (mirrors /graph/distances response):\n")
    print("  " + json.dumps(distances, indent=4).replace("\n", "\n  "))

    print("\n  Live server check:")
    try_live_endpoint(seed, max_hops)
    print()
