"""
Rebuild the relation graph for the 'papers' project from the existing BM25 corpus.
Runs only relation extraction (no re-embedding). Uses the live Axon config + Gemini.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

# ── point at the installed axon package ────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))

from axon.config import AxonConfig  # noqa: E402
from axon.llm import OpenLLM as LLMClient  # noqa: E402

# ── Load config first so we can resolve the store path ──────────────────
cfg = AxonConfig.load()

# Derive BM25 dir from config: <store_root>/<project>/bm25_index
# Override with --bm25-dir CLI arg or AXON_BM25_DIR env var.
import argparse  # noqa: E402

_p = argparse.ArgumentParser(description="Rebuild relation graph from BM25 corpus.")
_p.add_argument("--bm25-dir", default=os.getenv("AXON_BM25_DIR", ""), help="Path to bm25_index dir")
_p.add_argument(
    "--project", default="", help="Project name under the store root (alternative to --bm25-dir)"
)
_args = _p.parse_args()

if _args.bm25_dir:
    BM25_DIR = pathlib.Path(_args.bm25_dir)
elif _args.project:
    BM25_DIR = pathlib.Path(cfg.store_root) / _args.project / "bm25_index"
else:
    raise SystemExit("ERROR: pass --bm25-dir <path> or --project <name>  (or set AXON_BM25_DIR)")

CORPUS_FILE = BM25_DIR / "bm25_corpus.json"
ENTITY_FILE = BM25_DIR / ".entity_graph.json"
RELATION_FILE = BM25_DIR / ".relation_graph.json"
llm = LLMClient(cfg)

# ── Load corpus ─────────────────────────────────────────────────────────
print("Loading BM25 corpus…")
corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))
print(f"  {len(corpus)} chunks")

# ── Load entity graph ────────────────────────────────────────────────────
entity_graph = json.loads(ENTITY_FILE.read_text(encoding="utf-8"))
entity_names = set(entity_graph.keys())
print(f"  {len(entity_names)} known entities")


# ── Build set of chunk IDs that contain at least 1 known entity mention ─
def chunk_has_entity(text: str) -> bool:
    tl = text.lower()
    return any(ent in tl for ent in entity_names)


relevant = [doc for doc in corpus if chunk_has_entity(doc.get("text", ""))]
print(f"  {len(relevant)} chunks with entity mentions → extracting relations")

# ── Relation extraction prompt (mirrors graph_rag.py) ────────────────────
SYSTEM = "You are a knowledge graph extraction specialist."


def extract_relations(text: str) -> list[dict]:
    prompt = (
        "Extract key relationships from the following text.\n"
        "For each relationship output one line:\n"
        "  SUBJECT | RELATION | OBJECT | one-sentence description | strength (1-10)\n"
        "Strength: 1=weak/incidental, 10=core/defining. "
        "No bullets or extra text. If no clear relationships, output nothing.\n\n" + text[:3000]
    )
    try:
        raw = llm.complete(prompt, system_prompt=SYSTEM)
    except Exception as e:
        print(f"    LLM error: {e}")
        return []
    triples = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            try:
                strength = max(1, min(10, int(parts[4]))) if len(parts) >= 5 else 5
            except (ValueError, IndexError):
                strength = 5
            triples.append(
                {
                    "subject": parts[0],
                    "relation": parts[1],
                    "object": parts[2],
                    "description": parts[3],
                    "strength": strength,
                }
            )
    return triples


# ── Build relation graph ─────────────────────────────────────────────────
relation_graph: dict[str, list] = {}

for i, doc in enumerate(relevant):
    doc_id = doc.get("id", f"chunk_{i}")
    text = doc.get("text", "")
    print(f"  [{i+1}/{len(relevant)}] {doc_id[:60]}…", end=" ", flush=True)
    triples = extract_relations(text)
    print(f"{len(triples)} relations")

    for t in triples:
        src = t["subject"].lower().strip()
        tgt = t["object"].lower().strip()
        if not src or not tgt:
            continue
        entry = {
            "target": tgt,
            "relation": t["relation"].strip(),
            "chunk_id": doc_id,
            "description": t["description"],
            "weight": t["strength"],
            "strength": t["strength"],
            "support_count": 1,
            "text_unit_ids": [doc_id],
        }
        if src not in relation_graph:
            relation_graph[src] = []
        # deduplicate
        existing = next(
            (
                e
                for e in relation_graph[src]
                if e["target"] == tgt and e["relation"] == entry["relation"]
            ),
            None,
        )
        if existing:
            existing["weight"] = existing.get("weight", 1) + t["strength"]
            existing["support_count"] += 1
            if doc_id not in existing["text_unit_ids"]:
                existing["text_unit_ids"].append(doc_id)
        else:
            relation_graph[src].append(entry)

total_edges = sum(len(v) for v in relation_graph.values())
print(f"\n✓ Extracted {total_edges} relation edges across {len(relation_graph)} source entities")

# ── Save ─────────────────────────────────────────────────────────────────
RELATION_FILE.write_text(json.dumps(relation_graph), encoding="utf-8")
print(f"✓ Saved to {RELATION_FILE}")

# ── Update degree in entity graph ─────────────────────────────────────────
for src, entries in relation_graph.items():
    if src in entity_graph and isinstance(entity_graph[src], dict):
        entity_graph[src]["degree"] = len(entries)
    for e in entries:
        tgt = e["target"]
        if tgt in entity_graph and isinstance(entity_graph[tgt], dict):
            entity_graph[tgt]["degree"] = entity_graph[tgt].get("degree", 0) + 1

ENTITY_FILE.write_text(json.dumps(entity_graph), encoding="utf-8")
print(f"✓ Updated entity degrees in {ENTITY_FILE}")
print("Done. Restart the Axon server to pick up the new relation graph.")
