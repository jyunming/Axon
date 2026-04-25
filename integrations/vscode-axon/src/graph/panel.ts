/**
 * AxonGraphPanel — VS Code WebviewPanel that renders the knowledge / code graph.
 */

import * as vscode from 'vscode';

import * as path from 'path';

import * as fs from 'fs';

import * as os from 'os';

import * as crypto from 'crypto';

import { state, GRAPH_ANSWER_TIMEOUT_MS } from '../shared';

import { httpGet, httpPost, parseJsonSafe, formatDetail, normalizeGraphPayload } from '../client/http';

/**
 * Synthesize an answer from raw search chunks using the Copilot LM API.
 * Called as a fallback when the Axon /query (Ollama) endpoint is unavailable.
 */

async function synthesizeWithCopilot(query: string, chunks: any[]): Promise<string> {
  try {
    const models = await vscode.lm.selectChatModels({ family: 'gpt-4o' });
    const model = models[0];
    if (!model) { return ''; }
    const contextText = chunks.slice(0, 10).map((c: any, i: number) => {
      const src = (c.metadata?.source || c.source || '').split(/[\\/]/).pop() || '';
      const text = (c.text || c.content || '').slice(0, 600);
      return `[${i + 1}] ${src}\n${text}`;
    }).join('\n\n');
    const messages = [
      vscode.LanguageModelChatMessage.User(
        `You are a helpful assistant. Answer the following question using ONLY the provided context chunks. ` +
        `Format your answer with markdown (use **bold**, \`code\`, headings, bullet lists where appropriate).\n\n` +
        `Question: ${query}\n\nContext:\n${contextText}`
      )
    ];
    const response = await model.sendRequest(messages, {}, new vscode.CancellationTokenSource().token);
    let answer = '';
    for await (const part of response.text) {
      answer += part;
    }
    return answer;
  } catch (err) {
    return '';
  }

}

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
  private _currentQuery: string = '';
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
    this._currentQuery = data.query;
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
    } else if (msg.command === 'openInBrowser') {
      this._openInBrowser();
    }
  }
  private async _openInBrowser(): Promise<void> {
    const query = this._currentQuery;
    if (!query) { return; }
    const config = vscode.workspace.getConfiguration('axon');
    const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
    const apiKey = config.get<string>('apiKey', '');
    try {
      const result = await httpPost(`${apiBase}/query/visualize`, { query }, apiKey, 60_000);
      if (result.status !== 200) {
        vscode.window.showErrorMessage(`Axon: /query/visualize failed (${result.status})`);
        return;
      }
      const tmpFile = path.join(os.tmpdir(), `axon_graph_${Date.now()}.html`);
      fs.writeFileSync(tmpFile, result.body, 'utf8');
      vscode.env.openExternal(vscode.Uri.file(tmpFile));
    } catch (err) {
      vscode.window.showErrorMessage(`Axon: Could not open graph in browser. ${err}`);
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
    // Read graph-panel.js from disk at render time — bypasses all WebView script caching.
    const nonce = crypto.randomBytes(16).toString('base64');
    const panelJsPath = vscode.Uri.joinPath(this._extensionUri, 'media', 'graph-panel.js').fsPath;
    const panelJs = fs.readFileSync(panelJsPath, 'utf8').replace(/<\/script>/gi, '<\\/script>');
    const forceGraphUri = this._panel.webview.asWebviewUri(
      vscode.Uri.joinPath(this._extensionUri, 'media', '3d-force-graph.min.js')
    );
    // cspSrc covers the external force-graph lib; nonce covers the inlined panel script.
    const cspSrc = this._panel.webview.cspSource;
    const csp = `default-src 'none'; script-src ${cspSrc} 'nonce-${nonce}'; style-src 'unsafe-inline';`;
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
  #tab-bar { display: flex; align-items: center; border-bottom: 1px solid var(--ax-border); background: var(--ax-header-bg); flex-shrink: 0; }
  .tab { padding: 6px 16px; font-size: 0.78em; cursor: pointer; border-bottom: 2px solid transparent; color: var(--ax-muted); user-select: none; }
  .tab:hover { color: var(--ax-fg); background: var(--ax-hover-bg); }
  .tab-active { color: var(--ax-fg); border-bottom-color: var(--ax-accent); }
  .tab-disabled { opacity: 0.38; cursor: not-allowed; }
  #open-in-browser { margin-left: auto; padding: 4px 10px; font-size: 0.75em; cursor: pointer; color: var(--ax-muted); user-select: none; white-space: nowrap; }
  #open-in-browser:hover { color: var(--ax-fg); background: var(--ax-hover-bg); }
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
    <div id="open-in-browser" title="Open this graph in your browser">↗ Open in browser</div>
  </div>
  <div id="graph-area">
    <div id="graph-kg"></div>
    <div id="graph-cg"></div>
    <div id="graph-placeholder">No graph data available.<br>Ingest with <code>graph_rag: true</code> or <code>code_graph: true</code>.</div>
  </div>

</div>

<!-- Data passed as non-executable JSON — exempt from script-src CSP -->

<script type="application/json" id="app-data">${dataJson}</script>

<!-- force-graph loaded as external URI; panel script inlined with nonce (no disk cache) -->

<script src="${forceGraphUri}"></script>

<script nonce="${nonce}">${panelJs}</script>

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
  query: string,
  options?: {
    /** Skip the /query call and synthesize from preloadedSources via Copilot instead. */
    skipAxonQuery?: boolean;
    /** Pre-fetched search chunks — skips the /search/raw call when provided. */
    preloadedSources?: any[];
  }

): Promise<'opened' | 'updated' | 'no_graph_available' | 'query_failed'> {
  const config = vscode.workspace.getConfiguration('axon');
  const apiBase = config.get<string>('apiBase', 'http://127.0.0.1:8000');
  const apiKey = config.get<string>('apiKey', '');
  const graphSynthesis = config.get<boolean>('graphSynthesis', true);
  const skipQuery = options?.skipAxonQuery ?? false;
  const preloadedSources = options?.preloadedSources;
  const panel = AxonGraphPanel.createOrReveal(context);
  panel.showLoading(query);
  try {
    const queryPromise = (graphSynthesis && !skipQuery)
      ? httpPost(`${apiBase}/query`, { query, discuss: false, raptor: false, graph_rag: false }, apiKey, GRAPH_ANSWER_TIMEOUT_MS)
      : Promise.resolve({ status: 0, body: '{}' });
    const searchPromise = preloadedSources != null
      ? Promise.resolve({ status: 200, body: JSON.stringify({ results: preloadedSources }) })
      : httpPost(`${apiBase}/search/raw`, { query }, apiKey, GRAPH_ANSWER_TIMEOUT_MS);
    const [querySettled, searchSettled, kgSettled, cgSettled] = await Promise.allSettled([
      queryPromise,
      searchPromise,
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
    if (!graphSynthesis || skipQuery) {
      // skipQuery: search tool path — Copilot synthesizes from preloaded chunks below
      queryFailed = skipQuery;
      answerText = '';
    } else if (querySettled.status === 'fulfilled') {
      queryStatus = querySettled.value.status;
      const answer = parseJsonSafe(querySettled.value.body);
      if (querySettled.value.status === 200) {
        answerText = answer.response || '';
      } else {
        queryFailed = true;
        answerText = '';  // will be filled by Copilot fallback below
      }
    } else {
      queryFailed = true;
      answerText = '';  // will be filled by Copilot fallback below
    }
    if (searchSettled.status === 'fulfilled') {
      searchStatus = searchSettled.value.status;
      const search = parseJsonSafe(searchSettled.value.body);
      if (searchSettled.value.status === 200) {
        sources = Array.isArray(search.results) ? search.results : [];
      }
    }
    // Copilot fallback: if Axon /query failed but we have search results, synthesize via Copilot LM
    if (queryFailed && sources.length > 0) {
      state.outputChannel.appendLine(`[axon.showGraph] /query unavailable — falling back to Copilot synthesis from ${sources.length} chunks`);
      answerText = await synthesizeWithCopilot(query, sources);
      if (!answerText) {
        answerText = '_Query unavailable and Copilot synthesis failed. Showing search results only._';
      }
      queryFailed = false;  // we recovered, don't mark as failed in return value
    } else if (queryFailed) {
      answerText = '_Query unavailable — no search results to synthesize from._';
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

