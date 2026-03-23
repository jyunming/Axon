"""Code symbol graph build and persistence (CodeGraphMixin)."""
from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("Axon")


class CodeGraphMixin:
    def _load_code_graph(self) -> dict:
        """Load code graph from disk. Returns empty graph if not found."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    return data
            except Exception:
                pass
        return {"nodes": {}, "edges": []}

    def _save_code_graph(self) -> None:
        """Persist code graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._code_graph), encoding="utf-8")

    def _build_code_graph_from_chunks(self, chunks: list[dict]) -> None:
        """Build/update code graph nodes and CONTAINS/IMPORTS edges from codebase chunks.

        Nodes:
        - File node  — one per unique file_path
        - Symbol node — one per (file_path, symbol_name) with a real symbol_type

        Edges:
        - CONTAINS : File → Symbol
        - IMPORTS  : File → File  (resolved from imports metadata)
        """
        nodes: dict = self._code_graph.setdefault("nodes", {})
        edges_list: list = self._code_graph.setdefault("edges", [])
        existing_edges: set = {(e["source"], e["target"], e["edge_type"]) for e in edges_list}

        file_nodes_seen: set = set()
        # Deferred IMPORTS: collected during pass 1, resolved in pass 2 after all
        # file nodes exist so forward-referenced files are resolvable.
        deferred_imports: list[tuple[str, str, str]] = []  # (file_node_id, stmt, chunk_id)

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if meta.get("source_class") != "code":
                continue

            file_path = meta.get("file_path") or meta.get("source", "")
            language = meta.get("language", "unknown")
            symbol_type = meta.get("symbol_type", "block")
            symbol_name = meta.get("symbol_name", "")
            chunk_id = chunk.get("id", "")
            file_node_id = file_path

            # ── File node ───────────────────────────────────────────────────
            if file_path and file_path not in file_nodes_seen:
                file_nodes_seen.add(file_path)
                if file_node_id not in nodes:
                    nodes[file_node_id] = {
                        "node_id": file_node_id,
                        "node_type": "file",
                        "name": os.path.basename(file_path),
                        "file_path": file_path,
                        "language": language,
                        "chunk_ids": [],
                        "signature": "",
                        "start_line": None,
                        "end_line": None,
                    }

            if file_path and chunk_id:
                cids = nodes[file_node_id]["chunk_ids"]
                if chunk_id not in cids:
                    cids.append(chunk_id)

            # ── Symbol node ──────────────────────────────────────────────────
            if symbol_type not in ("block", "") and symbol_name and file_path:
                sym_node_id = f"{file_path}::{symbol_name}"
                if sym_node_id not in nodes:
                    nodes[sym_node_id] = {
                        "node_id": sym_node_id,
                        "node_type": symbol_type,
                        "name": symbol_name,
                        "file_path": file_path,
                        "language": language,
                        "chunk_ids": [chunk_id] if chunk_id else [],
                        "signature": meta.get("signature", ""),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                    }
                else:
                    if chunk_id and chunk_id not in nodes[sym_node_id]["chunk_ids"]:
                        nodes[sym_node_id]["chunk_ids"].append(chunk_id)

                # CONTAINS edge: File → Symbol
                ek = (file_node_id, sym_node_id, "CONTAINS")
                if ek not in existing_edges and file_path:
                    edges_list.append(
                        {
                            "source": file_node_id,
                            "target": sym_node_id,
                            "edge_type": "CONTAINS",
                            "chunk_id": chunk_id,
                        }
                    )
                    existing_edges.add(ek)

            # ── IMPORTS edges ────────────────────────────────────────────────
            imports_raw = meta.get("imports", "")
            if isinstance(imports_raw, str):
                import_stmts = [s for s in imports_raw.split("|") if s.strip()]
            elif isinstance(imports_raw, list):
                import_stmts = imports_raw
            else:
                import_stmts = []

            for stmt in import_stmts:
                deferred_imports.append((file_node_id, stmt.strip(), chunk_id))

        # Pass 2: resolve IMPORTS edges now that all file nodes are present.
        for file_node_id, stmt, chunk_id in deferred_imports:
            target_file_id = self._resolve_import_to_file(stmt)
            if target_file_id and target_file_id != file_node_id:
                ek = (file_node_id, target_file_id, "IMPORTS")
                if ek not in existing_edges:
                    edges_list.append(
                        {
                            "source": file_node_id,
                            "target": target_file_id,
                            "edge_type": "IMPORTS",
                            "chunk_id": chunk_id,
                        }
                    )
                    existing_edges.add(ek)

    def _resolve_import_to_file(self, stmt: str) -> str | None:
        """Resolve an import statement to a file node_id in the code graph, or None."""
        m = re.match(r"^from\s+([\w.]+)\s+import", stmt)
        if not m:
            m = re.match(r"^import\s+([\w.]+)", stmt)
        if not m:
            return None
        module = m.group(1)
        # e.g. "axon.splitters" → look for file_path ending in axon/splitters.py
        module_rel = module.replace(".", "/") + ".py"
        for node_id, node in self._code_graph.get("nodes", {}).items():
            if node.get("node_type") == "file":
                fp = node.get("file_path", "").replace("\\", "/")
                if fp.endswith(module_rel):
                    return node_id
        return None
