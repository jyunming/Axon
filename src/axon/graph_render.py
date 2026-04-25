"""Graph payload build and HTML render helpers (GraphRenderMixin)."""


from __future__ import annotations

import logging
import pathlib
import tempfile
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("Axon")


if TYPE_CHECKING:
    pass


class GraphRenderMixin:
    if TYPE_CHECKING:
        from axon.vector_store import OpenVectorStore

        _community_levels: dict[int, dict[str, Any]]
        _entity_graph: dict[str, dict]
        _relation_graph: dict[str, list[dict]]
        _code_graph: dict[str, Any]
        _VIZ_TYPE_COLORS: dict[str, str]
        vector_store: OpenVectorStore

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
                    "chunk_ids": node.get("chunk_ids", []),
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
    #node-info {{
      display: none; position: fixed; right: 14px; top: 14px; z-index: 30;
      width: 280px; max-height: 70vh; overflow-y: auto;
      padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.14); border-radius: 12px;
      background: rgba(8,12,20,0.92); backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.32);
      color: #e8edf7; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 12px;
    }}
  </style>
  <script src="https://unpkg.com/three"></script>
  <script src="https://unpkg.com/3d-force-graph"></script>

</head>

<body>
  <div class="hud">
    <h1>GraphRAG 3D Viewer</h1>
    <p>Left drag rotates &nbsp;·&nbsp; Right drag pans &nbsp;·&nbsp; Scroll zooms.</p>
    <p>Hover a node for details &nbsp;·&nbsp; Click to focus camera and show info.</p>
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
  <div id="node-info"></div>
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
      .linkLabel(link => `<div><b>${{esc(link.relation || 'relation')}}</b><br>${{esc(link.description || '')}}</div>`)
      .linkWidth(link => link.width || 1)
      .linkOpacity(0.45)
      .linkDirectionalParticles(link => Math.min(4, Math.max(1, Math.round((link.value || 1) / 10))))
      .linkDirectionalParticleSpeed(0.004)
      .linkDirectionalParticleWidth(2)
      .d3Force('charge').strength(-140);
    Graph.d3Force('link').distance(90);
    Graph.onNodeClick(node => {{
      const distance = 120;
      const x = Number.isFinite(node.x) ? node.x : 0;
      const y = Number.isFinite(node.y) ? node.y : 0;
      const z = Number.isFinite(node.z) ? node.z : 0;
      const radius = Math.hypot(x, y, z);
      const safeRadius = radius > 0 ? radius : 1;
      const distRatio = 1 + distance / safeRadius;
      Graph.cameraPosition(
        {{ x: x * distRatio, y: y * distRatio, z: z * distRatio }},
        {{ x, y, z }}, 1400
      );
      showNodeInfo(node);
    }});
    Graph.controls().autoRotate = false;
    Graph.controls().enableDamping = true;
    Graph.controls().dampingFactor = 0.12;
    function esc(s) {{
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }}
    function showNodeInfo(n) {{
      var panel = document.getElementById('node-info');
      var ev = Array.isArray(n.evidence) ? n.evidence : [];
      var evHtml = ev.length > 0
        ? '<div style="margin-top:8px;font-size:11px;opacity:0.7;text-transform:uppercase;letter-spacing:0.05em">Evidence (' + ev.length + ')</div>' +
          ev.map(function(e) {{
            var src = esc(e.source || '');
            var ln = e.start_line || 1;
            var exc = esc((e.excerpt || '').slice(0, 160));
            return '<div style="margin-top:6px;padding:4px 6px;background:rgba(255,255,255,0.05);border-radius:3px">' +
              '<div style="color:#4ec9b0;font-size:11px">' + (src.split('/').pop() || src) + ':' + ln + '</div>' +
              '<div style="font-size:11px;opacity:0.65;margin-top:2px">' + exc + '</div></div>';
          }}).join('')
        : '<div style="font-size:11px;opacity:0.5;font-style:italic;margin-top:6px">No evidence available</div>';
      panel.innerHTML =
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">' +
          '<b style="color:#9cdcfe">' + esc(n.name || n.id || '') + '</b>' +
          '<span style="cursor:pointer;opacity:0.6" onclick="document.getElementById(\'node-info\').style.display=\'none\'">✕</span>' +
        '</div>' +
        '<div style="font-size:11px;opacity:0.7">Type: ' + esc(n.type || 'UNKNOWN') + '</div>' +
        (n.description ? '<div style="font-size:11px;opacity:0.8;margin-top:4px">' + esc((n.description || '').slice(0, 200)) + '</div>' : '') +
        evHtml;
      panel.style.display = 'block';
    }}
  </script>

</body>

</html>

"""

    @staticmethod
    def render_query_graph_html(
        query: str,
        answer: str,
        sources: list,
        kg: dict,
        cg: dict,
    ) -> str:
        """Render a rich query-aware graph page as a standalone browser HTML string.
        Produces the same layout as the VS Code graph panel (left: query/answer/sources;
        right: KG + code-graph tabs with hit-node highlighting) but as a self-contained
        HTML file that works in any browser without VS Code.
        Args:
            query: The user query string shown at the top of the left panel.
            answer: The LLM answer rendered as Markdown in the left panel.
            sources: List of chunk dicts (with ``id``, ``text``, ``metadata`` keys)
                     used to highlight matching graph nodes and populate citations.
            kg: Knowledge-graph payload from :meth:`build_graph_payload`.
            cg: Code-graph payload from :meth:`build_code_graph_payload`.
        Returns:
            Self-contained HTML string — write to a file or serve directly.
        """
        import json as _json

        # Try to load the shared graph-panel.js from the VS Code extension media/ dir.
        # Falls back to a minimal inline implementation if the file is not found.
        panel_js: str | None = None
        _media_candidates = [
            # Installed from the repo root (dev environment)
            pathlib.Path(__file__).parent.parent.parent
            / "integrations"
            / "vscode-axon"
            / "media"
            / "graph-panel.js",
        ]
        for _p in _media_candidates:
            if _p.is_file():
                try:
                    raw = _p.read_text(encoding="utf-8")
                    # Replace `var vscode = acquireVsCodeApi();` with a browser shim.
                    # The shim intercepts openFile messages and shows a file-path banner
                    # instead of asking VS Code to open an editor tab.
                    raw = raw.replace(
                        "var vscode = acquireVsCodeApi();",
                        (
                            "var vscode = { postMessage: function(msg) {\n"
                            "  if (msg && msg.command === 'openFile') {\n"
                            "    var b = document.getElementById('browser-open-banner');\n"
                            "    if (b) {\n"
                            "      b.textContent = (msg.path || '') + (msg.line ? ':' + msg.line : '');\n"
                            "      b.style.display = 'block';\n"
                            "      setTimeout(function(){b.style.display='none';}, 4000);\n"
                            "    }\n"
                            "  }\n"
                            "} };"
                        ),
                    )
                    panel_js = raw
                except Exception:
                    pass
                break
        data = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "knowledgeGraph": kg,
            "codeGraph": cg,
        }
        data_json = _json.dumps(data, ensure_ascii=False).replace("</script>", "<\\/script>")
        if panel_js is not None:
            # Full-fidelity path: use the same graph-panel.js as VS Code, with CDN libs.
            panel_js_escaped = panel_js.replace("</script>", "<\\/script>")
            return f"""<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1">

<title>Axon — {_json.dumps(query[:60])}</title>

<style>
  :root {{
    --ax-bg: #1e1e1e; --ax-fg: #d4d4d4; --ax-muted: #888888;
    --ax-border: #333333; --ax-header-bg: #252526; --ax-hover-bg: #2a2d2e;
    --ax-accent: #569cd6; --ax-link: #4ec9b0; --ax-query: #9cdcfe;
    /* VS Code variable aliases so graph-panel.js theme lookups resolve */
    --vscode-editor-background: #1e1e1e; --vscode-editor-foreground: #d4d4d4;
    --vscode-descriptionForeground: #888888; --vscode-panel-border: #333333;
    --vscode-editorGroupHeader-tabsBackground: #252526;
    --vscode-list-hoverBackground: #2a2d2e; --vscode-focusBorder: #569cd6;
    --vscode-textLink-foreground: #4ec9b0; --vscode-textPreformat-foreground: #9cdcfe;
    --vscode-charts-blue: #569cd6; --vscode-charts-lines: #3c4a5a;
    --vscode-sideBar-background: #252526; --vscode-font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --vscode-font-size: 13px; --vscode-font-weight: 400;
    --vscode-disabledForeground: #555555;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--ax-bg); color: var(--ax-fg);
    font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif);
    font-size: 13px; height: 100vh; display: flex; overflow: hidden;
  }}
  #left {{ width: 35%; min-width: 260px; display: flex; flex-direction: column; border-right: 1px solid var(--ax-border); overflow: hidden; }}
  #right {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
  #tab-bar {{ display: flex; border-bottom: 1px solid var(--ax-border); background: var(--ax-header-bg); flex-shrink: 0; }}
  .tab {{ padding: 6px 16px; font-size: 0.78em; cursor: pointer; border-bottom: 2px solid transparent; color: var(--ax-muted); user-select: none; }}
  .tab:hover {{ color: var(--ax-fg); background: var(--ax-hover-bg); }}
  .tab-active {{ color: var(--ax-fg); border-bottom-color: var(--ax-accent); }}
  .tab-disabled {{ opacity: 0.38; cursor: not-allowed; }}
  #graph-area {{ flex: 1; position: relative; overflow: hidden; }}
  #graph-kg, #graph-cg {{ position: absolute; inset: 0; display: none; }}
  #graph-placeholder {{ position: absolute; inset: 0; display: none; align-items: center; justify-content: center; color: var(--ax-muted); font-style: italic; padding: 24px; text-align: center; }}
  .graph-tooltip {{ position: absolute; display: none; background: var(--ax-header-bg); color: var(--ax-fg); padding: 6px 10px; border-radius: 4px; font-size: 0.78em; max-width: 260px; pointer-events: none; border: 1px solid var(--ax-border); z-index: 10; word-break: break-word; }}
  #query-text {{ padding: 12px 16px; font-size: 0.85em; color: var(--ax-query); border-bottom: 1px solid var(--ax-border); font-weight: 600; word-break: break-word; }}
  #answer-text {{ padding: 12px 16px; font-size: 0.82em; line-height: 1.5; overflow-y: auto; flex: 1; border-bottom: 1px solid var(--ax-border); }}
  #citations {{ overflow-y: auto; max-height: 200px; padding: 8px; }}
  #citations-heading {{ padding: 6px 16px; font-size: 0.75em; color: var(--ax-muted); text-transform: uppercase; letter-spacing: 0.05em; background: var(--ax-header-bg); border-bottom: 1px solid var(--ax-border); }}
  .citation {{ padding: 6px 10px; cursor: default; border-bottom: 1px solid var(--ax-border); font-size: 0.78em; }}
  .citation:hover {{ background: var(--ax-hover-bg); }}
  .cite-num {{ color: var(--ax-accent); margin-right: 4px; }}
  .cite-src {{ color: var(--ax-link); font-weight: 500; }}
  .cite-text {{ color: var(--ax-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  #browser-open-banner {{
    display: none; position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(30,30,30,0.95); border: 1px solid #569cd6; color: #4ec9b0;
    padding: 6px 14px; border-radius: 6px; font-size: 12px; z-index: 40;
    white-space: nowrap; max-width: 80vw; overflow: hidden; text-overflow: ellipsis;
  }}

</style>

</head>

<body>

<div id="left">
  <div id="query-text"></div>
  <div id="answer-text"></div>
  <div id="citations-heading">Sources</div>
  <div id="citations"></div>

</div>

<div id="right">
  <div id="tab-bar">
    <div id="tab-kg" class="tab">Knowledge Graph</div>
    <div id="tab-cg" class="tab">Code Graph</div>
  </div>
  <div id="graph-area">
    <div id="graph-kg"></div>
    <div id="graph-cg"></div>
    <div id="graph-placeholder">No graph data available.<br>Ingest with <code>graph_rag: true</code> or <code>code_graph: true</code>.</div>
  </div>

</div>

<div id="browser-open-banner"></div>

<script type="application/json" id="app-data">{data_json}</script>

<script src="https://unpkg.com/three"></script>

<script src="https://unpkg.com/3d-force-graph"></script>

<script>{panel_js_escaped}</script>

</body>

</html>"""
        # Fallback: minimal inline implementation (no graph-panel.js found).
        n_kg = len(kg.get("nodes", []))
        n_cg = len(cg.get("nodes", []))
        kg_json = _json.dumps(kg, ensure_ascii=False).replace("</script>", "<\\/script>")
        cg_json = _json.dumps(cg, ensure_ascii=False).replace("</script>", "<\\/script>")
        import html as _html_mod

        sources_html = "".join(
            f'<div style="padding:5px 10px;border-bottom:1px solid #333;font-size:11px">'
            f'<span style="color:#4ec9b0">[{i+1}] '
            f'{_html_mod.escape((s.get("metadata") or {}).get("source", "") or s.get("source", ""))}'
            f'</span><div style="color:#888;margin-top:2px">'
            f'{_html_mod.escape((s.get("text","") or "")[:120])}…</div></div>'
            for i, s in enumerate(sources)
        )
        answer_esc = _html_mod.escape(answer or "*(no answer)*")
        query_esc = _html_mod.escape(query)
        return f"""<!DOCTYPE html>

<html lang="en"><head><meta charset="UTF-8">

<title>Axon Query — {query_esc[:60]}</title>

<style>
  body {{ margin:0; background:#1e1e1e; color:#d4d4d4; font-family:sans-serif; display:flex; height:100vh; overflow:hidden; }}
  #left {{ width:35%; border-right:1px solid #333; display:flex; flex-direction:column; overflow:hidden; }}
  #right {{ flex:1; display:flex; flex-direction:column; }}
  #query-text {{ padding:12px 16px; color:#9cdcfe; font-weight:600; border-bottom:1px solid #333; font-size:13px; }}
  #answer-text {{ padding:12px 16px; font-size:12px; line-height:1.55; overflow-y:auto; flex:1; border-bottom:1px solid #333; white-space:pre-wrap; }}
  #graph {{ flex:1; }}

</style>

</head><body>

<div id="left">
  <div id="query-text">{query_esc}</div>
  <div id="answer-text">{answer_esc}</div>
  <div style="padding:6px 16px;font-size:10px;color:#888;text-transform:uppercase;background:#252526;border-bottom:1px solid #333">Sources ({len(sources)})</div>
  <div style="overflow-y:auto;max-height:200px">{sources_html}</div>

</div>

<div id="right">
  <div style="padding:8px 16px;background:#252526;border-bottom:1px solid #333;font-size:11px;color:#888">
    Knowledge graph: {n_kg} nodes &nbsp;|&nbsp; Code graph: {n_cg} nodes
  </div>
  <div id="graph"></div>

</div>

<script src="https://unpkg.com/three"></script>

<script src="https://unpkg.com/3d-force-graph"></script>

<script>

var kg = {kg_json};

var graphData = kg.nodes && kg.nodes.length > 0 ? kg : {cg_json};

var elem = document.getElementById('graph');

if (graphData.nodes && graphData.nodes.length > 0) {{
  var Graph = ForceGraph3D()(elem)
    .graphData(graphData).backgroundColor('#1e1e1e')
    .nodeLabel(function(n) {{ return n.tooltip || n.name || n.id; }})
    .nodeColor(function(n) {{ return n.color || '#569cd6'; }})
    .nodeVal(function(n) {{ return n.val || 4; }})
    .linkWidth(function(l) {{ return l.width || 1; }}).linkOpacity(0.45);
  Graph.d3Force('charge').strength(-140);
  Graph.d3Force('link').distance(90);

}}

</script></body></html>"""

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
