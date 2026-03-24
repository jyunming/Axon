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

  /* ── Left panel: query / answer / citations ───────────────────────── */
  document.getElementById('query-text').textContent = 'Q: ' + DATA.query;

  var answerEl = document.getElementById('answer-text');
  answerEl.innerHTML = (DATA.answer || '(no answer)')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');

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

    var tooltip = document.createElement('div');
    tooltip.className = 'graph-tooltip';
    el.appendChild(tooltip);

    var graph = ForceGraph3D()(el)
      .graphData({ nodes: nodes, links: links })
      .backgroundColor(theme.background)
      .nodeLabel(function (n) { return n.tooltip || n.name || n.id || ''; })
      .nodeColor(function (n) { return n.color || theme.node; })
      .nodeVal(function (n) { return Math.max(1, n.val || 1); })
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
    return instance;
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
