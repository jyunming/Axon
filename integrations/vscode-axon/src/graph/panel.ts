/**
 * AxonGraphPanel — VS Code WebviewPanel that renders the knowledge / code graph.
 */
import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';

import { state, GRAPH_ANSWER_TIMEOUT_MS } from '../shared';
import { httpGet, httpPost, parseJsonSafe, formatDetail, normalizeGraphPayload } from '../client/http';

export interface GraphViewPayload {
  query: string;
  answer: string;
  sources: Array<{ id: string; source?: string; start_line?: number; text?: string; metadata?: Record<string, any>; content?: string }>;
  knowledgeGraph: { nodes: any[]; links: any[] };
  codeGraph: { nodes: any[]; links: any[] };
  meta: { created_at: string; query_status: number; graph_source: string };
}

export class AxonGraphPanel {
  static currentPanel: AxonGraphPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _extensionUri: vscode.Uri;
  private _disposables: vscode.Disposable[] = [];

  static createOrReveal(context: vscode.ExtensionContext): AxonGraphPanel {
    if (AxonGraphPanel.currentPanel) {
      AxonGraphPanel.currentPanel._panel.reveal(vscode.ViewColumn.Beside);
      return AxonGraphPanel.currentPanel;
    }
    const mediaUri = vscode.Uri.joinPath(context.extensionUri, 'media');
    const panel = vscode.window.createWebviewPanel(
      'axonGraph', 'Axon Graph', vscode.ViewColumn.Beside,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [mediaUri],
      }
    );
    AxonGraphPanel.currentPanel = new AxonGraphPanel(panel, context.extensionUri);
    return AxonGraphPanel.currentPanel;
  }

  private _disposed = false;

  private constructor(panel: vscode.WebviewPanel, extensionUri: vscode.Uri) {
    this._panel = panel;
    this._extensionUri = extensionUri;
    this._panel.onDidDispose(() => this.dispose(), null, this._disposables);
    this._panel.webview.onDidReceiveMessage(
      (msg: any) => this._handleMessage(msg),
      null, this._disposables
    );
    this._panel.webview.html = this._loadingHtml('Axon Graph');
  }

  update(data: GraphViewPayload) {
    this._panel.title = `Axon: ${data.query.slice(0, 40)}`;
    this._panel.webview.html = this._buildHtml(data);
  }

  showLoading(query: string) {
    this._panel.title = `Axon: ${query.slice(0, 40)}…`;
    this._panel.webview.html = this._loadingHtml(query);
  }

  private _handleMessage(msg: any) {
    if (msg.command === 'openFile') {
      const uri = vscode.Uri.file(msg.path);
      vscode.window.showTextDocument(uri, {
        selection: new vscode.Range(
          Math.max(0, (msg.line || 1) - 1), 0,
          Math.max(0, (msg.line || 1) - 1), 0
        )
      });
    }
  }

  private _loadingHtml(query: string): string {
    const csp = `default-src 'none'; style-src 'unsafe-inline';`;
    return `<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="Content-Security-Policy" content="${csp}"></head><body style="background:var(--vscode-editor-background,#1e1e1e);color:var(--vscode-editor-foreground,#cccccc);font-family:var(--vscode-font-family,sans-serif);display:flex;align-items:center;justify-content:center;height:100vh;margin:0;">
      <div style="text-align:center">
        <div style="font-size:2em;margin-bottom:1em">⟳</div>
        <div>Loading graph for: <em>${escapeHtml(query)}</em></div>
      </div>
    </body></html>`;
  }

  private _buildHtml(data: GraphViewPayload): string {
    // Serialize data for the webview — injected via <script type="application/json">,
    // which the browser treats as opaque data (never executed), so it is NOT subject
    // to script-src CSP. This eliminates the need for 'unsafe-inline'.
    const dataJson = JSON.stringify(data).replace(/<\/script>/gi, '<\\/script>');

    const forceGraphUri = this._panel.webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', '3d-force-graph.min.js')
    );
    const panelJsUri = this._panel.webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', 'graph-panel.js')
    );

    // No inline scripts → no 'unsafe-inline' needed.  Just cspSource covers both external files.
    const cspSrc = this._panel.webview.cspSource;
    const csp = `default-src 'none'; script-src ${cspSrc}; style-src 'unsafe-inline';`;

    return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<style>
  :root {
    --ax-bg: var(--vscode-editor-background, #1e1e1e);
    --ax-fg: var(--vscode-editor-foreground, #d4d4d4);
    --ax-muted: var(--vscode-descriptionForeground, #888888);
    --ax-border: var(--vscode-panel-border, #333333);
    --ax-header-bg: var(--vscode-editorGroupHeader-tabsBackground, #252526);
    --ax-hover-bg: var(--vscode-list-hoverBackground, #2a2d2e);
    --ax-accent: var(--vscode-focusBorder, #569cd6);
    --ax-link: var(--vscode-textLink-foreground, #4ec9b0);
    --ax-query: var(--vscode-textPreformat-foreground, #9cdcfe);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--ax-bg);
    color: var(--ax-fg);
    font-family: var(--vscode-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif);
    font-size: var(--vscode-font-size, 13px);
    font-weight: var(--vscode-font-weight, 400);
    height: 100vh;
    display: flex;
    overflow: hidden;
  }
  #left { width: 35%; min-width: 260px; display: flex; flex-direction: column; border-right: 1px solid var(--ax-border); overflow: hidden; }
  #right { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  #tab-bar { display: flex; border-bottom: 1px solid var(--ax-border); background: var(--ax-header-bg); flex-shrink: 0; }
  .tab { padding: 6px 16px; font-size: 0.78em; cursor: pointer; border-bottom: 2px solid transparent; color: var(--ax-muted); user-select: none; }
  .tab:hover { color: var(--ax-fg); background: var(--ax-hover-bg); }
  .tab-active { color: var(--ax-fg); border-bottom-color: var(--ax-accent); }
  .tab-disabled { opacity: 0.38; cursor: not-allowed; }
  #graph-area { flex: 1; position: relative; overflow: hidden; }
  #graph-kg, #graph-cg { position: absolute; inset: 0; display: none; }
  #graph-placeholder { position: absolute; inset: 0; display: none; align-items: center; justify-content: center; color: var(--ax-muted); font-style: italic; padding: 24px; text-align: center; }
  #graph-placeholder code { color: var(--ax-query); }
  .graph-tooltip { position: absolute; display: none; background: var(--ax-header-bg); color: var(--ax-fg); padding: 6px 10px; border-radius: 4px; font-size: 0.78em; max-width: 260px; pointer-events: none; border: 1px solid var(--ax-border); z-index: 10; word-break: break-word; }
  #query-text { padding: 12px 16px; font-size: 0.85em; color: var(--ax-query); border-bottom: 1px solid var(--ax-border); font-weight: 600; word-break: break-word; }
  #answer-text { padding: 12px 16px; font-size: 0.82em; line-height: 1.5; overflow-y: auto; flex: 1; border-bottom: 1px solid var(--ax-border); }
  #citations { overflow-y: auto; max-height: 200px; padding: 8px; }
  #citations-heading { padding: 6px 16px; font-size: 0.75em; color: var(--ax-muted); text-transform: uppercase; letter-spacing: 0.05em; background: var(--ax-header-bg); border-bottom: 1px solid var(--ax-border); }
  .citation { padding: 6px 10px; cursor: pointer; border-bottom: 1px solid var(--ax-border); font-size: 0.78em; }
  .citation:hover { background: var(--ax-hover-bg); }
  .cite-num { color: var(--ax-accent); margin-right: 4px; }
  .cite-src { color: var(--ax-link); font-weight: 500; }
  .cite-text { color: var(--ax-muted); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  body.vscode-high-contrast #left,
  body.vscode-high-contrast #tab-bar {
    border-color: var(--vscode-contrastBorder, var(--ax-border));
  }
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
<!-- Data passed as non-executable JSON — exempt from script-src CSP -->
<script type="application/json" id="app-data">${dataJson}</script>
<!-- Two external scripts only — no inline JS, satisfies strict script-src CSP -->
<script src="${forceGraphUri}"></script>
<script src="${panelJsUri}"></script>
</body>
</html>`;
  }

  dispose() {
    if (this._disposed) { return; }
    this._disposed = true;
    AxonGraphPanel.currentPanel = undefined;
    this._panel.dispose();
    this._disposables.forEach(d => d.dispose());
    this._disposables = [];
  }
}

function escapeHtml(str: string): string {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function saveGraphSnapshot(payload: GraphViewPayload): void {
  try {
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const queryHash = Buffer.from(payload.query).toString('base64').replace(/[^a-zA-Z0-9]/g, '').slice(0, 12);
    const snapshotDir = path.join(os.homedir(), '.axon', 'cache', 'graphs', `${ts}_${queryHash}`);
    fs.mkdirSync(snapshotDir, { recursive: true });
    fs.writeFileSync(path.join(snapshotDir, 'payload.json'), JSON.stringify(payload, null, 2));
    fs.writeFileSync(path.join(snapshotDir, 'meta.json'), JSON.stringify({
      timestamp: new Date().toISOString(),
      query_hash: queryHash,
      vscode_version: vscode.version,
    }, null, 2));
    state.outputChannel.appendLine(`[axon.showGraph] snapshot saved → ${snapshotDir}`);
  } catch (err) {
    state.outputChannel.appendLine(`[axon.showGraph] snapshot save failed: ${err}`);
  }
}

export async function showGraphForQuery(
  context: vscode.ExtensionContext,
  query: string
): Promise<'opened' | 'updated' | 'no_graph_available' | 'query_failed'> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');
  const graphSynthesis = config.get<boolean>('graphSynthesis', true);

  const panel = AxonGraphPanel.createOrReveal(context);
  panel.showLoading(query);

  try {
    const queryPromise = graphSynthesis
      ? httpPost(`${apiBase}/query`, { query, discuss: false }, apiKey, GRAPH_ANSWER_TIMEOUT_MS)
      : Promise.resolve({ status: 0, body: '{}' });
    const [querySettled, searchSettled, kgSettled, cgSettled] = await Promise.allSettled([
      queryPromise,
      httpPost(`${apiBase}/search/raw`, { query }, apiKey, GRAPH_ANSWER_TIMEOUT_MS),
      httpGet(`${apiBase}/graph/data`, apiKey),
      httpGet(`${apiBase}/code-graph/data`, apiKey),
    ]);

    let queryFailed = false;
    let answerText = '';
    let sources: any[] = [];
    let knowledgeGraph: { nodes: any[]; links: any[] } = { nodes: [], links: [] };
    let codeGraph: { nodes: any[]; links: any[] } = { nodes: [], links: [] };
    let queryStatus = 0;
    let searchStatus = 0;
    let kgStatus = 0;
    let cgStatus = 0;

    if (!graphSynthesis) {
      answerText = '';
    } else if (querySettled.status === 'fulfilled') {
      queryStatus = querySettled.value.status;
      const answer = parseJsonSafe(querySettled.value.body);
      if (querySettled.value.status === 200) {
        answerText = answer.response || '';
      } else {
        queryFailed = true;
        answerText = `(Query unavailable: ${formatDetail(answer, querySettled.value.body)})`;
      }
    } else {
      queryFailed = true;
      answerText = `(Query unavailable: ${String(querySettled.reason)})`;
    }

    if (searchSettled.status === 'fulfilled') {
      searchStatus = searchSettled.value.status;
      const search = parseJsonSafe(searchSettled.value.body);
      if (searchSettled.value.status === 200) {
        sources = Array.isArray(search.results) ? search.results : [];
      }
    }

    if (kgSettled.status === 'fulfilled') {
      kgStatus = kgSettled.value.status;
      if (kgSettled.value.status === 200) {
        knowledgeGraph = normalizeGraphPayload(parseJsonSafe(kgSettled.value.body));
      }
    }

    if (cgSettled.status === 'fulfilled') {
      cgStatus = cgSettled.value.status;
      if (cgSettled.value.status === 200) {
        codeGraph = normalizeGraphPayload(parseJsonSafe(cgSettled.value.body));
      }
    }

    const kgNodes = Array.isArray(knowledgeGraph.nodes) ? knowledgeGraph.nodes.length : 0;
    const cgNodes = Array.isArray(codeGraph.nodes) ? codeGraph.nodes.length : 0;
    state.outputChannel.appendLine(
      `[axon.showGraph] query="${query}" statuses query=${queryStatus} search=${searchStatus} kg=${kgStatus} cg=${cgStatus} nodes kg=${kgNodes} cg=${cgNodes}`
    );

    const payload: GraphViewPayload = {
      query,
      answer: answerText,
      sources,
      knowledgeGraph,
      codeGraph,
      meta: {
        created_at: new Date().toISOString(),
        query_status: queryStatus,
        graph_source: kgNodes > 0 ? 'knowledge' : cgNodes > 0 ? 'code' : 'none',
      },
    };

    panel.update(payload);
    saveGraphSnapshot(payload);

    if (kgNodes > 0 || cgNodes > 0) {
      return 'opened';
    }
    return queryFailed ? 'query_failed' : 'no_graph_available';
  } catch (err) {
    vscode.window.showErrorMessage(`Axon graph panel error: ${err}`);
    panel.dispose();
    return 'query_failed';
  }
}

export async function showGraphForSelection(context: vscode.ExtensionContext): Promise<void> {
  const editor = vscode.window.activeTextEditor;
  const query = editor?.document.getText(editor.selection).trim()
             || await vscode.window.showInputBox({ prompt: 'Enter query for Axon graph' });
  if (!query) { return; }
  await showGraphForQuery(context, query);
}
