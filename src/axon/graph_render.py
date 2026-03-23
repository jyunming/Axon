"""Graph payload build and HTML render helpers (GraphRenderMixin)."""
from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import TYPE_CHECKING

logger = logging.getLogger("Axon")

if TYPE_CHECKING:
    pass


class GraphRenderMixin:
    def build_graph_payload(self) -> dict:
        """Return a renderer-neutral graph payload normalised from internal graph state.

        The payload shape is::

            {
                "nodes": [{"id", "name", "label", "type", "color", "val",
                           "chunk_count", "degree", "community", "description",
                           "tooltip",
                           "evidence": [{"chunk_id", "source", "start_line", "excerpt"}, ...]
                           }, ...],
                "links": [{"source", "target", "label", "relation",
                           "description", "value", "width"}, ...]
            }

        ``evidence`` is populated from the vector store for each chunk ID
        referenced by the node.  It may be empty if the store is unavailable or
        no chunks have been ingested yet.

        This method separates graph extraction from rendering.  Feed the result
        to :meth:`export_graph_html` or any other renderer.
        """
        from html import escape

        # community_levels level-0 schema: {entity -> community_id (int)}
        entity_to_community: dict[str, int] = {}
        if self._community_levels:
            for entity, cid in self._community_levels.get(0, {}).items():
                try:
                    entity_to_community[entity] = int(cid)
                except (TypeError, ValueError):
                    pass

        def _tooltip(name: str, node: dict, community: int | None) -> str:
            desc = (node.get("description") or "").strip()
            desc = escape(desc[:220]) if desc else "No description"
            ntype = escape(node.get("type") or "UNKNOWN")
            chunk_count = len(node.get("chunk_ids", []))
            degree = node.get("degree", 0)
            comm = "None" if community is None else str(community)
            return (
                f"<div style='max-width:320px'>"
                f"<div><b>{escape(name)}</b></div>"
                f"<div><b>Type:</b> {ntype}</div>"
                f"<div><b>Chunks:</b> {chunk_count}</div>"
                f"<div><b>Degree:</b> {degree}</div>"
                f"<div><b>Community:</b> {comm}</div>"
                f"<div style='margin-top:6px'>{desc}</div>"
                f"</div>"
            )

        # Build a chunk_id → metadata lookup for evidence population.
        all_chunk_ids: list[str] = []
        for _node in self._entity_graph.values():
            if isinstance(_node, dict):
                all_chunk_ids.extend(_node.get("chunk_ids", []))
        chunk_meta_lookup: dict[str, dict] = {}
        if all_chunk_ids and hasattr(self, "vector_store"):
            try:
                for _doc in self.vector_store.get_by_ids(list(dict.fromkeys(all_chunk_ids))):
                    _cid = _doc.get("id") or _doc.get("chunk_id", "")
                    if _cid:
                        chunk_meta_lookup[_cid] = _doc.get("metadata", _doc)
            except Exception:
                pass

        nodes: list[dict] = []
        node_ids: set[str] = set()
        for name, node in self._entity_graph.items():
            if not isinstance(node, dict):
                continue
            community = entity_to_community.get(name)
            chunk_count = len(node.get("chunk_ids", []))
            evidence = [
                {
                    "chunk_id": cid,
                    "source": meta.get("source", ""),
                    "start_line": meta.get("start_line"),
                    "excerpt": (meta.get("text") or meta.get("page_content") or "")[:200],
                }
                for cid in node.get("chunk_ids", [])
                if (meta := chunk_meta_lookup.get(cid)) is not None
            ]
            nodes.append(
                {
                    "id": name,
                    "name": name,
                    "label": name[:24],
                    "type": node.get("type") or "UNKNOWN",
                    "color": self._VIZ_TYPE_COLORS.get(node.get("type") or "UNKNOWN", "#aec7e8"),
                    "val": 4 + min(chunk_count, 18),
                    "chunk_count": chunk_count,
                    "degree": node.get("degree", 0),
                    "community": community,
                    "description": (node.get("description") or "")[:220],
                    "tooltip": _tooltip(name, node, community),
                    "evidence": evidence,
                }
            )
            node_ids.add(name)

        links: list[dict] = []
        seen_edges: set[tuple] = set()
        for src, rels in self._relation_graph.items():
            if src not in node_ids:
                continue
            for rel in rels:
                if not isinstance(rel, dict):
                    continue
                tgt = rel.get("target") or rel.get("object", "")
                if not tgt or tgt not in node_ids:
                    continue
                relation = rel.get("relation", "")
                key = (src, tgt, relation)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                strength = float(rel.get("weight") or rel.get("strength") or 5)
                links.append(
                    {
                        "source": src,
                        "target": tgt,
                        "label": relation[:32],
                        "relation": relation,
                        "description": rel.get("description", ""),
                        "value": strength,
                        "width": 1 + strength / 8,
                    }
                )

        return {"nodes": nodes, "links": links}

    def build_code_graph_payload(self) -> dict:
        """Return the code structure graph as {nodes, links} for VS Code webview.

        Node types: ``file``, ``class``, ``function``, ``method``, ``module``.
        Edge types: ``CONTAINS`` (file→symbol), ``IMPORTS`` (file→file).
        Returns ``{"nodes": [], "links": []}`` when no code graph has been built.
        """
        _COLORS = {
            "file": "#4ec9b0",
            "module": "#569cd6",
            "class": "#c586c0",
            "function": "#dcdcaa",
            "method": "#dcdcaa",
        }
        nodes: list[dict] = []
        for node_id, node in self._code_graph.get("nodes", {}).items():
            ntype = node.get("node_type", "unknown")
            sig = node.get("signature", "")
            label = node.get("name", node_id)
            nodes.append(
                {
                    "id": node_id,
                    "name": label,
                    "label": label[:28],
                    "type": ntype,
                    "color": _COLORS.get(ntype, "#888888"),
                    "val": 5 if ntype == "file" else 3,
                    "file_path": node.get("file_path", ""),
                    "start_line": node.get("start_line") or 1,
                    "chunk_ids": node.get("chunk_ids", []),
                    "tooltip": f"[{ntype}] {label}" + (f"\n{sig}" if sig else ""),
                }
            )
        links: list[dict] = []
        for edge in self._code_graph.get("edges", []):
            links.append(
                {
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "label": edge.get("edge_type", ""),
                    "edge_type": edge.get("edge_type", ""),
                    "value": 1,
                    "width": 1,
                }
            )
        return {"nodes": nodes, "links": links}

    @staticmethod
    def _render_graph_html(graph: dict) -> str:
        """Render a graph payload (from :meth:`build_graph_payload`) as a 3D HTML viewer.

        The viewer loads three.js and 3d-force-graph from unpkg.com CDN — requires
        internet access to render.  The HTML file itself needs no server.
        """
        import json as _json

        # Escape </script> so a crafted entity string cannot break out of the script context.
        data_json = _json.dumps(graph, ensure_ascii=False).replace("</script>", "<\\/script>")
        n_nodes = len(graph["nodes"])
        n_links = len(graph["links"])
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>GraphRAG 3D Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body, #graph {{
      margin: 0; width: 100%; height: 100%;
      overflow: hidden;
      background: #0b1020; color: #e8edf7;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .hud {{
      position: fixed; top: 14px; left: 14px; z-index: 20;
      max-width: 360px; padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.14); border-radius: 12px;
      background: rgba(8,12,20,0.84); backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }}
    .hud h1 {{ margin: 0 0 8px; font-size: 14px; }}
    .hud p  {{ margin: 4px 0; font-size: 12px; line-height: 1.45; opacity: 0.9; }}
    .legend {{ display: grid; grid-template-columns: auto 1fr;
               gap: 6px 10px; margin-top: 10px; font-size: 11px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 999px; margin-top: 3px; }}
  </style>
  <script src="https://unpkg.com/three"></script>
  <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
  <div class="hud">
    <h1>GraphRAG 3D Viewer</h1>
    <p>Left drag rotates &nbsp;·&nbsp; Right drag pans &nbsp;·&nbsp; Scroll zooms.</p>
    <p>Hover a node for details &nbsp;·&nbsp; Click to focus camera.</p>
    <p>Nodes: {n_nodes} &nbsp;|&nbsp; Edges: {n_links}</p>
    <div class="legend">
      <div class="swatch" style="background:#4e79a7"></div><div>PERSON</div>
      <div class="swatch" style="background:#f28e2b"></div><div>ORGANIZATION</div>
      <div class="swatch" style="background:#59a14f"></div><div>GEO</div>
      <div class="swatch" style="background:#e15759"></div><div>EVENT</div>
      <div class="swatch" style="background:#76b7b2"></div><div>CONCEPT</div>
      <div class="swatch" style="background:#edc948"></div><div>PRODUCT</div>
      <div class="swatch" style="background:#bab0ab"></div><div>UNKNOWN</div>
    </div>
  </div>
  <div id="graph"></div>
  <script>
    const graphData = {data_json};
    const elem = document.getElementById('graph');
    const Graph = ForceGraph3D()(elem)
      .graphData(graphData)
      .backgroundColor('#0b1020')
      .nodeLabel(node => node.tooltip)
      .nodeColor(node => node.color)
      .nodeVal(node => node.val)
      .nodeOpacity(0.95)
      .linkLabel(link => `<div><b>${{link.relation || 'relation'}}</b><br>${{link.description || ''}}</div>`)
      .linkWidth(link => link.width || 1)
      .linkOpacity(0.45)
      .linkDirectionalParticles(link => Math.min(4, Math.max(1, Math.round((link.value || 1) / 10))))
      .linkDirectionalParticleSpeed(0.004)
      .linkDirectionalParticleWidth(2)
      .d3Force('charge').strength(-140);
    Graph.d3Force('link').distance(90);
    Graph.onNodeClick(node => {{
      const distance = 120;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
      Graph.cameraPosition(
        {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
        node, 1400
      );
    }});
    Graph.controls().autoRotate = false;
    Graph.controls().enableDamping = true;
    Graph.controls().dampingFactor = 0.12;
  </script>
</body>
</html>
"""

    def export_graph_html(
        self,
        path: str | None = None,
        json_path: str | None = None,
        open_browser: bool = True,
    ) -> str:
        """Export the entity–relation graph as a self-contained 3D interactive HTML viewer.

        Normalises internal graph state into a renderer-neutral payload via
        :meth:`build_graph_payload`, then renders it with three.js + 3d-force-graph.

        Args:
            path: File path to write the HTML to.  Defaults to a temp file when
                  *open_browser* is True and no path is provided.
            json_path: Optional path to also write the normalised graph JSON payload.
            open_browser: If True (default), open the generated HTML in the default
                          web browser immediately after writing.

        Returns:
            The rendered HTML string.
        """
        import json as _json

        graph = self.build_graph_payload()
        html = self._render_graph_html(graph)

        if json_path:
            pathlib.Path(json_path).write_text(
                _json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Graph JSON payload saved to %s (%d nodes, %d edges)",
                json_path,
                len(graph["nodes"]),
                len(graph["links"]),
            )

        if path:
            pathlib.Path(path).write_text(html, encoding="utf-8")
            logger.info(
                "Graph visualization saved to %s (%d nodes, %d edges)",
                path,
                len(graph["nodes"]),
                len(graph["links"]),
            )
            out_path = path
        elif open_browser:
            # Write to a temp file so the browser can load it
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", prefix="axon_graph_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            out_path = tmp.name
            logger.info("Graph visualization written to temp file %s", out_path)
        else:
            out_path = None

        if open_browser and out_path:
            import webbrowser

            webbrowser.open(f"file://{pathlib.Path(out_path).resolve()}")
            logger.info("Opened graph visualization in default browser.")

        return html
