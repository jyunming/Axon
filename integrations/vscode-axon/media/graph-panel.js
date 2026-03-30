/**
 * graph-panel.js — Axon VS Code webview panel
 *
 * Loaded as an external <script src="..."> — no inline JS, so the strict
 * "script-src ${cspSource}" CSP is satisfied without 'unsafe-inline'.
 *
 * Data arrives via <script type="application/json" id="app-data"> which is
 * never executed by the browser and therefore not subject to script-src CSP.
 *
 * Depends on: ForceGraph3D (3d-force-graph.min.js loaded before this file).
 */
(function () {
  'use strict';

  /* ── Read data injected by the extension ──────────────────────────── */
  var rawEl = document.getElementById('app-data');
  if (!rawEl) { return; }
  var DATA = JSON.parse(rawEl.textContent);

  var vscode = acquireVsCodeApi();

  function openFile(fp, ln) {
    vscode.postMessage({ command: 'openFile', path: fp, line: ln });
  }

  function getThemeColor(name, fallback) {
    var value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
  }

  function getGraphTheme() {
    var chartBlue = getThemeColor('--vscode-charts-blue', '');
    return {
      background: getThemeColor('--vscode-editor-background', '#1e1e1e'),
      node: chartBlue || getThemeColor('--vscode-focusBorder', '#569cd6'),
      linkImport: chartBlue || '#569cd6',
      linkContains: getThemeColor('--vscode-descriptionForeground', '#555555'),
      linkOther: getThemeColor('--vscode-charts-lines', '#3c4a5a')
    };
  }

  /* Scale a hex color's brightness by factor (0–1 dims, >1 brightens). */
  function dimHex(hex, factor) {
    var c = (hex || '#888888').replace(/^#/, '');
    if (c.length === 3) { c = c[0]+c[0]+c[1]+c[1]+c[2]+c[2]; }
    if (c.length !== 6) { return hex; }
    var toHex = function (v) { var s = Math.min(255, Math.round(v)).toString(16); return s.length < 2 ? '0'+s : s; };
    return '#' + toHex(parseInt(c.substr(0,2),16)*factor)
               + toHex(parseInt(c.substr(2,2),16)*factor)
               + toHex(parseInt(c.substr(4,2),16)*factor);
  }


  /* ── Left panel: query / answer / citations ───────────────────────── */
  document.getElementById('query-text').textContent = 'Q: ' + DATA.query;

  /* Lightweight Markdown → HTML (no external lib) */
  function renderMarkdown(md) {
    if (!md) { return '<em style="opacity:0.5">(no answer)</em>'; }
    var esc = function (s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    };
    var lines = md.split('\n');
    var out = [];
    var inCode = false, codeLang = '', codeBuf = [];
    var inList = false, listOl = false;
    var flushList = function () {
      if (!inList) { return; }
      out.push(listOl ? '</ol>' : '</ul>');
      inList = false;
    };
    for (var i = 0; i < lines.length; i++) {
      var l = lines[i];
      /* ── fenced code block ── */
      if (/^```/.test(l)) {
        if (!inCode) {
          flushList();
          inCode = true;
          codeLang = l.slice(3).trim();
          codeBuf = [];
        } else {
          out.push('<pre style="background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);border-radius:3px;padding:8px 10px;overflow-x:auto;margin:6px 0;font-size:11.5px"><code>' + esc(codeBuf.join('\n')) + '</code></pre>');
          inCode = false; codeLang = ''; codeBuf = [];
        }
        continue;
      }
      if (inCode) { codeBuf.push(l); continue; }
      /* ── blank line ── */
      if (/^\s*$/.test(l)) { flushList(); out.push('<div style="height:6px"></div>'); continue; }
      /* ── headings ── */
      var hm = l.match(/^(#{1,4})\s+(.*)/);
      if (hm) {
        flushList();
        var hLevel = hm[1].length;
        var hSizes = ['15px','13.5px','12.5px','12px'];
        out.push('<div style="font-size:' + hSizes[hLevel-1] + ';font-weight:600;margin:8px 0 3px;color:var(--vscode-editor-foreground,#d4d4d4)">' + inlinemd(esc(hm[2])) + '</div>');
        continue;
      }
      /* ── horizontal rule ── */
      if (/^---+$/.test(l.trim())) { flushList(); out.push('<hr style="border:none;border-top:1px solid rgba(255,255,255,0.12);margin:6px 0">'); continue; }
      /* ── unordered list ── */
      var ulm = l.match(/^(\s*)[-*+]\s+(.*)/);
      if (ulm) {
        if (!inList || listOl) { flushList(); out.push('<ul style="margin:4px 0 4px 16px;padding:0">'); inList = true; listOl = false; }
        out.push('<li style="margin:2px 0">' + inlinemd(esc(ulm[2])) + '</li>');
        continue;
      }
      /* ── ordered list ── */
      var olm = l.match(/^(\s*)\d+[.)]\s+(.*)/);
      if (olm) {
        if (!inList || !listOl) { flushList(); out.push('<ol style="margin:4px 0 4px 16px;padding:0">'); inList = true; listOl = true; }
        out.push('<li style="margin:2px 0">' + inlinemd(esc(olm[2])) + '</li>');
        continue;
      }
      /* ── blockquote ── */
      var bqm = l.match(/^>\s?(.*)/);
      if (bqm) { flushList(); out.push('<div style="border-left:3px solid rgba(255,255,255,0.25);padding-left:8px;color:rgba(255,255,255,0.6);margin:3px 0">' + inlinemd(esc(bqm[1])) + '</div>'); continue; }
      /* ── paragraph ── */
      flushList();
      out.push('<div style="margin:2px 0">' + inlinemd(esc(l)) + '</div>');
    }
    flushList();
    if (inCode) { out.push('<pre><code>' + esc(codeBuf.join('\n')) + '</code></pre>'); }
    return out.join('');
  }

  function inlinemd(s) {
    /* inline code */
    s = s.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1);padding:1px 4px;border-radius:2px;font-size:11.5px">$1</code>');
    /* bold+italic */
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    /* bold */
    s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    s = s.replace(/__(.+?)__/g, '<strong>$1</strong>');
    /* italic */
    s = s.replace(/\*(.+?)\*/g, '<em>$1</em>');
    s = s.replace(/_(.+?)_/g, '<em>$1</em>');
    /* strikethrough */
    s = s.replace(/~~(.+?)~~/g, '<del>$1</del>');
    /* links — already HTML-escaped so &amp; is already gone; match raw angle-bracket safe form */
    s = s.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a style="color:var(--vscode-textLink-foreground,#3794ff)" href="$2">$1</a>');
    return s;
  }

  var answerEl = document.getElementById('answer-text');
  answerEl.style.lineHeight = '1.55';
  answerEl.innerHTML = renderMarkdown(DATA.answer);

  var citContainer = document.getElementById('citations');
  (DATA.sources || []).forEach(function (s, i) {
    var src  = (s.metadata && s.metadata.source) || s.source || '';
    var line = (s.metadata && s.metadata.start_line) || s.start_line || 1;
    var text = ((s.text || s.content || '').slice(0, 120))
               .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    var div = document.createElement('div');
    div.className = 'citation';
    div.innerHTML =
      '<span class="cite-num">[' + (i + 1) + ']</span>' +
      '<span class="cite-src">' +
        escHtml(src.split(/[\\/]/).pop() || src) + ':' + line +
      '</span>' +
      '<div class="cite-text">' + text + '&hellip;</div>';
    div.addEventListener('click', function () { openFile(src, line); });
    citContainer.appendChild(div);
  });

  /* ── Right panel: tabs ────────────────────────────────────────────── */
  var kg = DATA.knowledgeGraph || { nodes: [], links: [] };
  var cg = DATA.codeGraph      || { nodes: [], links: [] };

  /* ── Cross-reference graph nodes with retrieved search results ─────── */
  (function () {
    function normSrc(p) { return (p || '').replace(/\\/g, '/'); }
    var hitFilePaths = new Set();
    var hitChunkIds  = new Set();
    (DATA.sources || []).forEach(function (s) {
      var src = normSrc((s.metadata && s.metadata.source) || s.source || '');
      if (src) { hitFilePaths.add(src); }
      if (s.id) { hitChunkIds.add(s.id); }
    });

    if (!hitFilePaths.size && !hitChunkIds.size) { return; }

    function isCodeHit(n) {
      var fp = normSrc(n.file_path || '');
      if (!fp) { return false; }
      if (hitFilePaths.has(fp)) { return true; }
      var hit = false;
      hitFilePaths.forEach(function (src) {
        if (!hit && (fp.endsWith('/' + src) || src.endsWith('/' + fp.split('/').pop()))) {
          hit = true;
        }
      });
      return hit;
    }

    function isKgHit(n) {
      var chunkIds = n.chunk_ids || [];
      if (chunkIds.some(function (id) { return hitChunkIds.has(id); })) { return true; }
      var ev = Array.isArray(n.evidence) ? n.evidence : [];
      return ev.some(function (e) {
        var src = normSrc(e.source || '');
        if (!src) { return false; }
        if (hitFilePaths.has(src)) { return true; }
        var found = false;
        hitFilePaths.forEach(function (h) {
          if (!found && src.endsWith('/' + h.split('/').pop())) { found = true; }
        });
        return found;
      });
    }

    kg.nodes.forEach(function (n) { if (isKgHit(n))   { n._hit = true; } });
    cg.nodes.forEach(function (n) { if (isCodeHit(n)) { n._hit = true; } });

    /* Mark 1st-degree neighbours of hit nodes */
    function markNeighbors(nodes, links) {
      var nodeById = {};
      nodes.forEach(function (n) { nodeById[n.id] = n; });
      var hitIds = new Set();
      nodes.forEach(function (n) { if (n._hit) { hitIds.add(n.id); } });
      if (!hitIds.size) { return; }
      links.forEach(function (l) {
        var s = (l.source && typeof l.source === 'object') ? l.source.id : l.source;
        var t = (l.target && typeof l.target === 'object') ? l.target.id : l.target;
        if (hitIds.has(s) && nodeById[t] && !nodeById[t]._hit) { nodeById[t]._neighbor = true; }
        if (hitIds.has(t) && nodeById[s] && !nodeById[s]._hit) { nodeById[s]._neighbor = true; }
      });
    }
    markNeighbors(kg.nodes, kg.links || []);
    markNeighbors(cg.nodes, cg.links || []);
  })();

  var kgHasData = Array.isArray(kg.nodes) && kg.nodes.length > 0;
  var cgHasData = Array.isArray(cg.nodes) && cg.nodes.length > 0;

  /* Tab buttons */
  var tabKg = document.getElementById('tab-kg');
  var tabCg = document.getElementById('tab-cg');
  if (!kgHasData) { tabKg.classList.add('tab-disabled'); tabKg.title = 'No knowledge graph — ingest with graph_rag: true'; }
  if (!cgHasData) { tabCg.classList.add('tab-disabled'); tabCg.title = 'No code graph — ingest with code_graph: true'; }

  /* Graph containers */
  var kgContainer = document.getElementById('graph-kg');
  var cgContainer = document.getElementById('graph-cg');
  var placeholder = document.getElementById('graph-placeholder');

  /* Active graph instance (so we can resize it) */
  var activeGraphInstance = null;

  if (!kgHasData && !cgHasData) {
    placeholder.style.display = 'flex';
    return;
  }

  /* Default: prefer code graph when query looks code-related, otherwise KG.
     Falls back gracefully if the preferred graph has no data. */
  var CODE_KEYWORDS = /\b(code|file|function|class|method|module|import|def|src|api|endpoint|variable|const|interface|type|struct|object|library|package)\b/i;
  var queryLooksCode = CODE_KEYWORDS.test(DATA.query || '');
  var initialTab = (queryLooksCode && cgHasData) ? 'cg'
                 : kgHasData                     ? 'kg'
                 :                                 'cg';
  showTab(initialTab);

  tabKg.addEventListener('click', function () { if (kgHasData) { showTab('kg'); } });
  tabCg.addEventListener('click', function () { if (cgHasData) { showTab('cg'); } });

  var openInBrowserBtn = document.getElementById('open-in-browser');
  if (openInBrowserBtn) {
    openInBrowserBtn.addEventListener('click', function () {
      vscode.postMessage({ command: 'openInBrowser' });
    });
  }

  window.addEventListener('resize', function () {
    var el = activeGraphInstance && activeGraphInstance._el;
    if (el && activeGraphInstance._graph) {
      activeGraphInstance._graph.width(el.clientWidth).height(el.clientHeight);
    }
  });

  /* ── Tab switching ─────────────────────────────────────────────────── */
  function showTab(tab) {
    tabKg.classList.toggle('tab-active', tab === 'kg');
    tabCg.classList.toggle('tab-active', tab === 'cg');
    kgContainer.style.display = tab === 'kg' ? 'block' : 'none';
    cgContainer.style.display = tab === 'cg' ? 'block' : 'none';

    // Dismiss any open KG node-detail panel — it does not belong on the CG tab
    // and would appear as an opaque dark overlay on the right side of the graph.
    if (detailPanel) {
      detailPanel.remove();
      detailPanel = null;
    }

    if (tab === 'kg' && !kgContainer._graphMounted && kgHasData) {
      kgContainer._graphMounted = true;
      activeGraphInstance = mountGraph(kgContainer, kg, 'knowledge');
    } else if (tab === 'cg' && !cgContainer._graphMounted && cgHasData) {
      cgContainer._graphMounted = true;
      activeGraphInstance = mountGraph(cgContainer, cg, 'code');
    }

    if (tab === 'kg' && kgContainer._graphMounted) {
      activeGraphInstance = kgContainer._instance;
    } else if (tab === 'cg' && cgContainer._graphMounted) {
      activeGraphInstance = cgContainer._instance;
    }

    // Resize the canvas on the next paint frame.
    // When display flips none→block the browser hasn't reflowed yet; reading
    // clientWidth synchronously can return a stale value so the WebGL viewport
    // only covers part of the container, leaving the rest black.
    var inst = activeGraphInstance;
    requestAnimationFrame(function () {
      if (inst && inst._graph && inst._el) {
        inst._graph.width(inst._el.clientWidth).height(inst._el.clientHeight);
      }
    });
  }

  /* ── Mount a ForceGraph3D into a container element ─────────────────── */
  function mountGraph(el, graphData, graphType) {
    var nodes = graphData.nodes.map(function (n) { return Object.assign({}, n); });
    var links = graphData.links.map(function (l) { return Object.assign({}, l); });
    var theme = getGraphTheme();
    var hasHits = nodes.some(function (n) { return n._hit; });

    var tooltip = document.createElement('div');
    tooltip.className = 'graph-tooltip';
    el.appendChild(tooltip);

    var hasNeighbors = nodes.some(function (n) { return n._neighbor; });

    var graph = ForceGraph3D()(el)
      .graphData({ nodes: nodes, links: links })
      .backgroundColor(theme.background)
      .nodeLabel(function (n) { return n.tooltip || n.name || n.id || ''; })
      .nodeColor(function (n) {
        var base = n.color || theme.node;
        if (!hasHits) { return base; }
        if (n._hit || n._neighbor) { return dimHex(base, 1.2); }
        return dimHex(base, 0.55);
      })
      .nodeVal(function (n) {
        if (n._hit)      { return Math.max(6,  (n.val || 1) * 3); }
        if (n._neighbor) { return Math.max(3,  (n.val || 1) * 2); }
        return hasHits ? Math.max(0.5, (n.val || 1) * 0.55) : Math.max(1, n.val || 1);
      })
      /* Saturn-style torus ring around hit/neighbor nodes */
      .nodeThreeObjectExtend(true)
      .nodeThreeObject(function (n) {
        if (!n._hit && !n._neighbor) { return null; }
        var T = window.THREE;
        if (!T) { return null; }
        var nv = n._hit ? Math.max(6, (n.val || 1) * 3.5)
                        : Math.max(3, (n.val || 1) * 2);
        /* ForceGraph3D default nodeRelSize = 4 */
        var sphereR = Math.cbrt(nv) * 4;
        var ringR  = n._hit ? sphereR * 1.75 : sphereR * 1.5;
        var tubeR  = n._hit ? sphereR * 0.22  : sphereR * 0.16;
        var tilt   = n._hit ? Math.PI / 5     : Math.PI / 4;
        var geo = new T.TorusGeometry(ringR, tubeR, 8, 32);
        var mat = new T.MeshLambertMaterial({
          color: n._hit ? 0xffcc00 : 0xff9f3a,
          transparent: true,
          opacity: n._hit ? 0.95 : 0.80
        });
        var mesh = new T.Mesh(geo, mat);
        mesh.rotation.x = tilt;
        return mesh;
      })
      .nodeOpacity(0.9)
      .linkLabel(function (l) { return l.label || l.edge_type || l.relation || ''; })
      .linkColor(function (l) {
        return (l.edge_type === 'IMPORTS') ? theme.linkImport
             : (l.edge_type === 'CONTAINS') ? theme.linkContains
             : theme.linkOther;
      })
      .linkWidth(function (l) { return l.width || 1; })
      .linkOpacity(0.6)
      .onNodeClick(function (n) { handleNodeClick(n, graphType); })
      .width(el.clientWidth)
      .height(el.clientHeight);

    var instance = { _el: el, _graph: graph };
    el._instance = instance;
    buildLegend(el, nodes, graphType, theme, hasHits, hasNeighbors);
    return instance;
  }

  /* ── Per-type legend overlay ────────────────────────────────────────── */
  function buildLegend(el, nodes, graphType, theme, hasHits, hasNeighbors) {
    /* Collect unique type → {color, total, hits, neighbors} */
    var typeMap = {};
    nodes.forEach(function (n) {
      var t = n.type || (graphType === 'code' ? 'node' : 'entity');
      if (!typeMap[t]) { typeMap[t] = { color: n.color || theme.node, total: 0, hits: 0, neighbors: 0 }; }
      typeMap[t].total++;
      if (n._hit)      { typeMap[t].hits++; }
      if (n._neighbor) { typeMap[t].neighbors++; }
    });
    var legend = document.createElement('div');
    legend.style.cssText =
      'position:absolute;top:8px;left:8px;z-index:10;pointer-events:none;' +
      'background:rgba(0,0,0,0.68);color:var(--vscode-editor-foreground,#d4d4d4);' +
      'font-size:11px;font-family:var(--vscode-font-family,sans-serif);' +
      'padding:7px 10px;border:1px solid rgba(255,255,255,0.12);border-radius:4px;min-width:140px;line-height:1;';
    var html = '<div style="color:rgba(255,255,255,0.45);font-size:9.5px;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:5px">Node types</div>';
    Object.keys(typeMap).sort().forEach(function (t) {
      var info   = typeMap[t];
      var isHit  = info.hits > 0;
      var isNear = !isHit && info.neighbors > 0;
      /* Dot color: 120% brightness for hit/neighbor, 40% dim for others */
      var dotColor = (hasHits && !isHit && !isNear) ? dimHex(info.color, 0.55) : dimHex(info.color, 1.2);
      /* Dot size: direct=12px, 1st connection=8px, others=4px */
      var dotSize  = isHit ? 12 : (isNear ? 8 : 4);
      /* Label opacity: 40% for unrelated types */
      var labelOpacity = (hasHits && !isHit && !isNear) ? '0.4' : '1';
      html +=
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">' +
          '<span style="width:12px;height:12px;flex-shrink:0;display:inline-flex;align-items:center;justify-content:center">' +
            '<span style="width:' + dotSize + 'px;height:' + dotSize + 'px;border-radius:50%;display:inline-block;background:' + dotColor + '"></span>' +
          '</span>' +
          '<span style="opacity:' + labelOpacity + '">' + escHtml(t) + ' · ' + info.total + '</span>' +
        '</div>';
    });
    legend.innerHTML = html;
    el.appendChild(legend);
  }

  /* ── Node click handler ────────────────────────────────────────────── */
  function handleNodeClick(n, graphType) {
    if (graphType === 'code') {
      /* Code graph nodes carry file_path + start_line directly */
      var fp = n.file_path || '';
      var ln = n.start_line || 1;
      if (fp) { openFile(fp, ln); }
      return;
    }

    /* Knowledge graph — prefer node.evidence (populated by build_graph_payload),
       fall back to citation-list lookup via chunk_ids for older payloads. */
    var ev = Array.isArray(n.evidence) ? n.evidence : [];
    if (ev.length > 0 && ev[0].source) {
      openFile(ev[0].source, ev[0].start_line || 1);
      showNodeDetail(n);
      return;
    }

    /* Legacy fallback: scan sources by chunk_id */
    var chunkIds = n.chunk_ids || [];
    if (!chunkIds.length) { return; }
    var found = null;
    for (var k = 0; k < DATA.sources.length; k++) {
      if (DATA.sources[k].id === chunkIds[0]) { found = DATA.sources[k]; break; }
    }
    if (!found) { return; }
    var fp2 = (found.metadata && found.metadata.source) ? found.metadata.source : '';
    var ln2 = (found.metadata && found.metadata.start_line) ? found.metadata.start_line : 1;
    if (fp2) { openFile(fp2, ln2); }
  }

  /* ── Node detail side panel ─────────────────────────────────────────── */
  var detailPanel = null;
  function showNodeDetail(n) {
    if (!detailPanel) {
      detailPanel = document.createElement('div');
      detailPanel.style.cssText =
        'position:fixed;right:0;top:0;bottom:0;width:280px;background:var(--vscode-sideBar-background,#252526);' +
        'border-left:1px solid var(--vscode-panel-border,#333);overflow-y:auto;padding:12px;font-size:0.78em;' +
        'color:var(--vscode-editor-foreground,#d4d4d4);z-index:20;';
      document.body.appendChild(detailPanel);
    }
    var ev = Array.isArray(n.evidence) ? n.evidence : [];
    var evHtml = ev.map(function (e, i) {
      var src = escHtml(e.source || '');
      var ln = e.start_line || 1;
      var exc = escHtml((e.excerpt || '').slice(0, 160));
      return '<div style="margin-bottom:8px;cursor:pointer;padding:4px;border:1px solid var(--vscode-panel-border,#333);border-radius:3px" ' +
             'onclick="(function(){var m=window.__axonOpenFile;if(m){m(' + JSON.stringify(e.source) + ',' + ln + ');}})();">' +
             '<div style="color:var(--vscode-textLink-foreground,#4ec9b0)">' + (src.split('/').pop() || src) + ':' + ln + '</div>' +
             '<div style="color:var(--vscode-descriptionForeground,#888);margin-top:2px">' + exc + '</div>' +
             '</div>';
    }).join('');
    detailPanel.innerHTML =
      '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">' +
        '<b style="color:var(--vscode-textPreformat-foreground,#9cdcfe)">' + escHtml(n.name || n.id || '') + '</b>' +
        '<span style="cursor:pointer;color:var(--vscode-descriptionForeground,#888)" onclick="this.parentNode.parentNode.remove();window.__axonDetailPanel=null">✕</span>' +
      '</div>' +
      '<div style="color:var(--vscode-descriptionForeground,#888);margin-bottom:4px">Type: ' + escHtml(n.type || '') + '</div>' +
      '<div style="color:var(--vscode-descriptionForeground,#888);margin-bottom:8px">' + escHtml((n.description || '').slice(0, 200)) + '</div>' +
      (ev.length ? '<div style="color:var(--vscode-descriptionForeground,#888);text-transform:uppercase;font-size:0.9em;margin-bottom:4px">Evidence (' + ev.length + ')</div>' + evHtml
                 : '<div style="color:var(--vscode-disabledForeground,#555);font-style:italic">No evidence available</div>');
    /* Expose openFile so inline onclick can call back */
    window.__axonOpenFile = openFile;
    window.__axonDetailPanel = detailPanel;
  }

  /* ── Utilities ─────────────────────────────────────────────────────── */
  function escHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
})();
